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

/// Whether the byte bounds share any bytes.
pub(crate) fn overlaps(a: &Range<usize>, b: &Range<usize>) -> bool { !a.is_empty() && !b.is_empty() && a.start < b.end && b.start < a.end }
