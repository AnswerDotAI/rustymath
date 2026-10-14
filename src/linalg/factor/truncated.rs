//! The truncated SVD, by Halko, Martinsson and Tropp's randomized range finder, on every platform.
use super::*;

/// Columns beyond `k` in `truncated_factor`'s test matrix. Halko, Martinsson and Tropp suggest about 10.
const OVERSAMPLE: usize = 10;
/// Power iterations in `truncated_factor`. Each multiplies by `aᴴ` and `a` once more, which separates slowly decaying singular values.
const POWERS: usize = 2;

/// `truncated_svd` for `T`: an approximation to the `k` largest singular values of the `m`×`n` matrix `a` and their singular vectors,
/// by Halko, Martinsson and Tropp's randomized range finder. `a` times `k + OVERSAMPLE` pseudorandom columns spans a space close to
/// that of the leading left singular vectors. `POWERS` power iterations refine it, with a thin QR factorization after each product.
/// The SVD of `a` projected onto that space gives the result. Where
/// `k + OVERSAMPLE` reaches `a`'s smaller side, the result is the thin SVD's first `k` columns. A matrix holding a NaN or an infinity
/// gives NaN for every result.
pub(super) fn truncated_factor<T: Tiled>(a: &[T], [m, n]: [usize; 2], k: usize) -> Svd<T>
where
    T::Real: Tiled<Real = T::Real>,
{
    assert_eq!(a.len(), m * n, "`a` must hold `m`×`n` items");
    let p = m.min(n);
    let (k, l) = (k.min(p), (k + OVERSAMPLE).min(p));
    if l == p { return leading(svd_factor(a, [m, n], true), k) }
    // A wide matrix works on its conjugate transpose, through a view, and swaps U and V at the end: aᴴ = V Σ Uᴴ.
    let wide = m < n;
    let (view, [rows, cols]) = if wide { (Matrix::rows(a, n).adjoint(), [n, m]) } else { (Matrix::rows(a, n), [m, n]) };
    let times = |a: Matrix<'_, T>, [r, inner]: [usize; 2], x: &[T]| T::multiply(a, Matrix::rows(x, l), [r, inner, l]);
    let basis = |y: Vec<T>, r: usize| qr_factor(&y, [r, l], false).0;
    let mut q = basis(times(view, [rows, cols], &test_matrix(cols * l)), rows);
    for _ in 0..POWERS { q = basis(times(view, [rows, cols], &basis(times(view.adjoint(), [cols, rows], &q), cols)), rows) }
    // B = Qᴴ times the view is `l`×`cols`. With B's SVD U_B Σ Vᴴ, the view is approximately (Q U_B) Σ Vᴴ.
    let b = T::multiply(Matrix::rows(&q, l).adjoint(), view, [l, rows, cols]);
    let Svd { u: ub, s, v, .. } = svd_factor(&b, [l, cols], true);
    let u = T::multiply(Matrix::rows(&q, l), Matrix::rows(&ub, l), [rows, l, l]);
    let (u, v) = if wide { (v, u) } else { (u, v) };
    leading(Svd::new([m, n], u, s, v), k)
}

/// The first `k` singular values of `svd`, with their singular vectors. A vector that keeps less than half its columns releases its
/// spare capacity.
fn leading<T: Scalar>(Svd { size: [m, n], mut u, mut s, mut v, .. }: Svd<T>, k: usize) -> Svd<T> {
    let p = s.len();
    for (x, rows) in [(&mut u, m), (&mut v, n)] {
        compact(x, [rows, k], p);
        if 2 * k < p { x.shrink_to_fit() }
    }
    s.truncate(k);
    Svd::new([m, n], u, s, v)
}

/// `len` numbers uniform from -1 to 1, from SplitMix64 with a fixed seed.
fn test_matrix<T: Scalar>(len: usize) -> Vec<T> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    (0..len).map(|_| {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let z = z ^ (z >> 31);
        T::from_real(<T::Real as crate::Float>::narrow((z >> 11) as f64 * 2f64.powi(-52) - 1.0))
    }).collect()
}
