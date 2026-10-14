//! Byte bounds for proving that NumPy callback ranges can run independently.
use std::ops::Range;

/// The bytes reached by a strided array, or `None` if its address arithmetic overflows. Empty arrays reach no bytes.
pub(crate) fn span(base: usize, item_bytes: usize, dims: &[(usize, isize)]) -> Option<Range<usize>> {
    if dims.iter().any(|&(n, _)| n == 0) { return Some(base..base) }
    let (mut lo, mut hi) = (base as i128, base as i128);
    for &(n, stride) in dims {
        let tail = ((n - 1) as i128).checked_mul(stride as i128)?;
        lo = lo.checked_add(tail.min(0))?;
        hi = hi.checked_add(tail.max(0))?;
    }
    hi = hi.checked_add(item_bytes as i128)?;
    Some(usize::try_from(lo).ok()?..usize::try_from(hi).ok()?)
}

/// The bytes reached by `n` items `stride` bytes apart, each a strided array as `span` takes, or `None` where two items could
/// share bytes or the address arithmetic overflows. Items are separate where the stride is at least one item's extent.
pub(crate) fn separate(base: usize, item_bytes: usize, dims: &[(usize, isize)], n: usize, stride: isize) -> Option<Range<usize>> {
    let one = span(base, item_bytes, dims)?;
    if !one.is_empty() && stride.unsigned_abs() < one.len() { return None }
    span(base, item_bytes, &[dims, &[(n, stride)]].concat())
}

/// Whether the byte bounds share any bytes.
pub(crate) fn overlaps(a: &Range<usize>, b: &Range<usize>) -> bool { !a.is_empty() && !b.is_empty() && a.start < b.end && b.start < a.end }
