//! Compact array construction and evenly spaced sequences.
use crate::array::Array;
use crate::buffer::{Element, View};
use crate::error::{Error, Kind};
use crate::layout::{Layout, generated_len};

/// Fills `layout` with the one item of `value`, retaining its dtype. A Length error unless `value` holds one item.
pub fn full(layout: Layout, value: View<'_>) -> Result<Array, Error> {
    if value.len() != 1 { return Err(Error::new(Kind::Length, "a fill value must have one item")); }
    let n = generated_len(layout.shape())?;
    let buffer = with_dtype!(value.dtype(), T => {
        let mut out = Vec::with_capacity(n);
        out.resize(n, T::of(value).unwrap()[0]);
        T::buffer(out)
    });
    Array::new(layout, buffer)
}

/// A matrix with `values` on diagonal `offset`, and the dtype's fill elsewhere. One value repeats; a vector must match the diagonal's length.
pub fn diagonal(values: View<'_>, shape: [usize; 2], offset: i64) -> Result<Array, Error> {
    let cols = shape[1];
    let (row, col, len) = crate::structure::diagonal_span(offset, shape);
    if values.len() != 1 && values.len() != len { return Err(Error::new(Kind::Length, "values must fill the diagonal, or hold one item")); }
    let n = generated_len(&shape)?;
    let buffer = with_dtype!(values.dtype(), T => {
        let values = T::of(values).expect("matching dtype");
        let mut out = Vec::with_capacity(n);
        out.resize(n, T::FILL);
        for i in 0..len { out[(row + i) * cols + col + i] = values[if values.len() == 1 { 0 } else { i }]; }
        T::buffer(out)
    });
    Array::new(shape.into(), buffer)
}

/// Keeps the upper or lower triangle of every matrix in `layout`'s last two axes, filling the other positions with the dtype's fill.
pub fn triangle(layout: Layout, values: View<'_>, offset: i64, upper: bool) -> Result<Array, Error> {
    let shape = layout.shape();
    if shape.len() < 2 { return Err(Error::new(Kind::Rank, "triangles need at least two axes")); }
    let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let buffer = with_dtype!(values.dtype(), T => {
        let data = T::of(values).expect("matching dtype");
        let mut out = Vec::with_capacity(data.len());
        if rows > 0 && cols > 0 {
            for (i, row) in data.chunks(cols).enumerate() {
                let end = (i as i128 % rows as i128 + offset as i128 + i128::from(!upper)).clamp(0, cols as i128) as usize;
                if upper { out.resize(out.len() + end, T::FILL); crate::gather::copied(&mut out, &row[end..]); }
                else { crate::gather::copied(&mut out, &row[..end]); out.resize(out.len() + cols - end, T::FILL); }
            }
        }
        T::buffer(out)
    });
    Array::new(layout, buffer)
}

/// Integers from `start` toward `stop`, excluding `stop`, separated by a nonzero `step`.
pub fn arange_int(start: i64, stop: i64, step: i64) -> Result<Vec<i64>, Error> {
    if step == 0 { return Err(Error::new(Kind::Domain, "a range's step can't be zero")); }
    let distance = if step > 0 { i128::from(stop) - i128::from(start) } else { i128::from(start) - i128::from(stop) };
    let width = i128::from(step).abs();
    let n = (distance.max(0) + width - 1) / width;
    let n = generated_len(&[usize::try_from(n).unwrap_or(usize::MAX)])?;
    Ok((0..n).map(|i| start.wrapping_add((i as i64).wrapping_mul(step))).collect())
}

/// Floats from `start` toward `stop`, excluding `stop`, separated by a finite nonzero `step`.
pub fn arange(start: f64, stop: f64, step: f64) -> Result<Vec<f64>, Error> {
    if !start.is_finite() || !stop.is_finite() || !step.is_finite() || step == 0.0 {
        return Err(Error::new(Kind::Domain, "a range needs finite bounds and a finite nonzero step"));
    }
    let n = ((stop - start) / step).ceil().max(0.0);
    let n = generated_len(&[n as usize])?;
    Ok((0..n).map(|i| start + i as f64 * step).collect())
}

/// `n` evenly spaced values from `start` to `stop`. With `endpoint`, the last item is `stop` when there is more than one item.
pub fn linspace(start: f64, stop: f64, n: usize, endpoint: bool) -> Result<Vec<f64>, Error> {
    generated_len(&[n])?;
    let step = (stop - start) / n.saturating_sub(usize::from(endpoint)).max(1) as f64;
    let mut out: Vec<_> = (0..n).map(|i| start + i as f64 * step).collect();
    if endpoint && n > 1 { out[n - 1] = stop; }
    Ok(out)
}

/// Powers of `base` at `n` evenly spaced exponents from `start` to `stop`.
pub fn logspace(start: f64, stop: f64, n: usize, endpoint: bool, base: f64) -> Result<Vec<f64>, Error> {
    let exponents = linspace(start, stop, n, endpoint)?;
    let mut out = Vec::with_capacity(n);
    crate::Binary::many(&crate::Pow, crate::Arg::One(base), crate::Arg::Many(&exponents), n, &mut out);
    Ok(out)
}
