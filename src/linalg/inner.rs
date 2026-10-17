//! Bounded inner products for native function pairs without a dedicated tiled kernel.
use crate::{agreement::{Agreement, Mapping}, buffer::{Buffer, View}, gather::Steps, layout::Layout, CHUNK};

/// An inner product of compact row-major matrices, with sizes `[m, k, n]` and `k > 0`. `g` applies the item function through the agreement; `fold` folds each term block from the right, with its previously folded right tail when given. Each block has shape `[1, depth, columns]` and at most [`CHUNK`] terms. A singleton without a tail keeps `g`'s result and never calls `fold`.
///
/// The callbacks choose their own native kernels and report failure as an error. They run at block boundaries, not per item. For non-associative functions, `fold` keeps right association across its tail, including converting a Boolean running result back to the input type when needed. Associative reductions may regroup their items. Frontends keep their own empty-contraction identities and optimized pairs.
pub fn inner_with<E>(a: View<'_>, b: View<'_>, [m, k, n]: [usize; 3],
    g: impl Fn(View<'_>, View<'_>, &Agreement) -> Result<Buffer, E>,
    fold: impl Fn(View<'_>, [usize; 3], Option<View<'_>>) -> Result<Buffer, E>,
) -> Result<Buffer, E> {
    assert!(k > 0, "an empty contraction needs the fold's identity");
    assert!(a.len() == m * k && b.len() == k * n, "the matrices must fill their dimensions");
    if m == 0 || n == 0 {
        let empty = Agreement { layout: Layout::EMPTY, len: 0, left: Mapping::Single, right: Mapping::Single };
        let terms = g(a.slice(0..0), b.slice(0..0), &empty)?;
        return if k == 1 { Ok(terms) } else { fold(terms.view(), [0, k, 0], None) };
    }
    let mut result: Option<Buffer> = None;
    let columns = n.min(CHUNK);
    for row in 0..m {
        for column in (0..n).step_by(columns) {
            let width = columns.min(n - column);
            let depth = (CHUNK / width).max(1);
            let mut end = k;
            let mut tail: Option<Buffer> = None;
            while end > 0 {
                let start = end.saturating_sub(depth);
                let rows = end - start;
                let right_items = b.slice(start * n + column..(end - 1) * n + column + width);
                let right = if width == n || rows == 1 { Mapping::Linear(1) } else { Mapping::Steps { base: 0, steps: vec![
                    Steps::Stride { start: 0, len: rows, step: n as isize }, Steps::Stride { start: 0, len: width, step: 1 },
                ] } };
                let agreement = Agreement {
                    layout: Layout::from([rows, width]), len: rows * width, left: Mapping::Linear(width), right,
                };
                let terms = g(a.slice(row * k + start..row * k + end), right_items, &agreement)?;
                let total = if rows == 1 && tail.is_none() { terms }
                    else { fold(terms.view(), [1, rows, width], tail.as_ref().map(Buffer::view))? };
                tail = Some(total);
                end = start;
            }
            let total = tail.expect("a nonempty contraction has a total");
            let output = result.get_or_insert_with(|| Buffer::with_capacity(total.dtype(), m * n));
            assert!(output.extend_from(total.view(), 0..width), "a native function keeps its output dtype");
        }
    }
    Ok(result.expect("a nonempty result has a buffer"))
}
