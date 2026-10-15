//! Linear algebra in any field, such as the exact rationals of `BigRational`: Gauss-Jordan row reduction with the solves, inverses and determinants built on it, and the permanent. Row reduction pivots on the first nonzero item of each column, which loses nothing in exact arithmetic. In floats it can lose all accuracy, so floats take [`lu`](super::lu) and the other decompositions, which pivot on the largest item. Each function has a `_with` form that calls `check` once for each column, or between groups of subsets for the permanent, and returns the check's error as soon as it gets one, as [`number_theory`](crate::number_theory)'s functions do.
use super::identity;
use crate::{BETWEEN_CHECKS, field::{negated, sum, times}, ok};
use num_traits::NumAssignRef;

/// A matrix in reduced row echelon form, as [`echelon`] gives it.
pub struct Echelon<T> {
    /// The nonzero rows. Each row's first nonzero item, its pivot, is 1, and is the only nonzero item in its column.
    pub rows: Vec<T>,
    /// The column of each row's pivot, in increasing order.
    pub pivots: Vec<usize>,
    /// The product of the pivots before each was scaled to 1, negated once for each swap of two rows. Where every one of the reduced columns has a pivot, it is the determinant of the matrix those columns form.
    pub determinant: T,
}

/// The reduced row echelon form of the `m`×`width` matrix `e`, reducing only its first `columns` columns.
pub fn echelon_with<T: Clone + NumAssignRef, E>(
    mut e: Vec<T>, [m, width]: [usize; 2], columns: usize, mut check: impl FnMut() -> Result<(), E>) -> Result<Echelon<T>, E> {
    assert_eq!(e.len(), m * width, "`e` must hold `m`×`width` items");
    let (mut pivots, mut determinant) = (Vec::new(), T::one());
    for c in 0..columns.min(width) {
        if pivots.len() == m { break; }
        check()?;
        let p = pivots.len();
        let Some(row) = (p..m).find(|&r| !e[r * width + c].is_zero()) else { continue };
        if row != p {
            for j in 0..width { e.swap(p * width + j, row * width + j) }
            determinant = negated(determinant);
        }
        let pivot = e[p * width + c].clone();
        for j in c..width { e[p * width + j] /= &pivot }
        determinant *= &pivot;
        for i in (0..m).filter(|&i| i != p) {
            let factor = e[i * width + c].clone();
            if factor.is_zero() { continue; }
            for j in c..width {
                let change = times(&factor, &e[p * width + j]);
                e[i * width + j] -= &change;
            }
        }
        pivots.push(c);
    }
    e.truncate(pivots.len() * width);
    Ok(Echelon { rows: e, pivots, determinant })
}

/// [`echelon_with`] with no check.
pub fn echelon<T: Clone + NumAssignRef>(e: Vec<T>, size: [usize; 2], columns: usize) -> Echelon<T> { let Ok(e) = echelon_with(e, size, columns, ok); e }

/// The determinant of the `n`×`n` matrix `a`.
pub fn determinant_with<T: Clone + NumAssignRef, E>(a: &[T], n: usize, check: impl FnMut() -> Result<(), E>) -> Result<T, E> {
    let e = echelon_with(a.to_vec(), [n, n], n, check)?;
    Ok(if e.pivots.len() < n { T::zero() } else { e.determinant })
}

/// [`determinant_with`] with no check.
pub fn determinant<T: Clone + NumAssignRef>(a: &[T], n: usize) -> T { let Ok(d) = determinant_with(a, n, ok); d }

/// The solution `x` of `c x = r` and the determinant of `c`, for the `n`×`n` matrix `c` and the `n`×`k` matrix `r`, from the echelon form of `[c r]`. `None` where `c` is singular.
fn square_with<T: Clone + NumAssignRef, E>(c: &[T], r: &[T], n: usize, k: usize, check: impl FnMut() -> Result<(), E>) -> Result<Option<(Vec<T>, T)>, E> {
    let joined = (0..n).flat_map(|i| c[i * n..(i + 1) * n].iter().chain(&r[i * k..(i + 1) * k]).cloned()).collect();
    let e = echelon_with(joined, [n, n + k], n, check)?;
    Ok((e.pivots.len() == n).then(|| ((0..n * k).map(|ij| e.rows[ij / k * (n + k) + n + ij % k].clone()).collect(), e.determinant)))
}

/// The least-squares solution `x` of `a x = b` with the least norm, for the real `m`×`n` matrix `a` and the `m`×`k` matrix `b`, as an `n`×`k` matrix. A square `a` of full rank gives the exact solution. Otherwise `x` comes from a rank factorization `a = C F`, where `C` holds `a`'s pivot columns and `F` the nonzero rows of its echelon form: `x = Fᵀ (F Fᵀ)⁻¹ (Cᵀ C)⁻¹ Cᵀ b`.
pub fn solve_with<T: Clone + NumAssignRef, E>(a: &[T], b: &[T], [m, n]: [usize; 2], k: usize, mut check: impl FnMut() -> Result<(), E>) -> Result<Vec<T>, E> {
    assert_eq!(a.len(), m * n, "`a` must hold `m`×`n` items");
    assert_eq!(b.len(), m * k, "`b` must hold `m`×`k` items");
    if m == n { if let Some((x, _)) = square_with(a, b, n, k, &mut check)? { return Ok(x); } }
    let Echelon { rows: f, pivots, .. } = echelon_with(a.to_vec(), [m, n], n, &mut check)?;
    let r = pivots.len();
    let c = |row: usize, i: usize| &a[row * n + pivots[i]];
    let gram: Vec<T> = (0..r * r).map(|ij| sum((0..m).map(|row| times(c(row, ij / r), c(row, ij % r))))).collect();
    let y: Vec<T> = (0..r * k).map(|ij| sum((0..m).map(|row| times(c(row, ij / k), &b[row * k + ij % k])))).collect();
    let z = square_with(&gram, &y, r, k, &mut check)?.expect("pivot columns are independent").0;
    let rows: Vec<T> = (0..r * r).map(|ij| sum((0..n).map(|j| times(&f[ij / r * n + j], &f[ij % r * n + j])))).collect();
    let w = square_with(&rows, &z, r, k, &mut check)?.expect("an echelon form's rows are independent").0;
    Ok((0..n * k).map(|ij| sum((0..r).map(|i| times(&f[i * n + ij / k], &w[i * k + ij % k])))).collect())
}

/// [`solve_with`] with no check.
pub fn solve<T: Clone + NumAssignRef>(a: &[T], b: &[T], size: [usize; 2], k: usize) -> Vec<T> { let Ok(x) = solve_with(a, b, size, k, ok); x }

/// The pseudoinverse of the real `m`×`n` matrix `a`, as an `n`×`m` matrix: its inverse where `a` is square and of full rank.
pub fn pinv_with<T: Clone + NumAssignRef, E>(a: &[T], [m, n]: [usize; 2], check: impl FnMut() -> Result<(), E>) -> Result<Vec<T>, E> {
    solve_with(a, &identity(m), [m, n], m, check)
}

/// [`pinv_with`] with no check.
pub fn pinv<T: Clone + NumAssignRef>(a: &[T], size: [usize; 2]) -> Vec<T> { let Ok(x) = pinv_with(a, size, ok); x }

/// The inverse of the `n`×`n` matrix `a` and its determinant, from one elimination, or `None` where `a` is singular or empty.
pub fn inverse_with<T: Clone + NumAssignRef, E>(a: &[T], n: usize, check: impl FnMut() -> Result<(), E>) -> Result<Option<(Vec<T>, T)>, E> {
    assert_eq!(a.len(), n * n, "`a` must hold `n`×`n` items");
    if n == 0 { return Ok(None); }
    square_with(a, &identity(n), n, n, check)
}

/// [`inverse_with`] with no check.
pub fn inverse<T: Clone + NumAssignRef>(a: &[T], n: usize) -> Option<(Vec<T>, T)> { let Ok(x) = inverse_with(a, n, ok); x }

/// The permanent of the `n`×`n` matrix `a`: the determinant's sum over permutations, with every sign positive. It follows Ryser's formula, a sum over the subsets of the columns that visits them in Gray code order, so each step adds or removes one column. That takes about 2ⁿ n operations. Panics if `n` is 64 or more.
pub fn permanent_with<T: Clone + NumAssignRef, E>(a: &[T], n: usize, mut check: impl FnMut() -> Result<(), E>) -> Result<T, E> {
    assert_eq!(a.len(), n * n, "`a` must hold `n`×`n` items");
    assert!(n < 64, "the permanent takes fewer than 64 rows");
    // Each row's sum over the columns in the current subset.
    let mut sums = vec![T::zero(); n];
    let (mut total, mut previous) = (T::zero(), 0u64);
    for s in 1..1u64 << n {
        if s % BETWEEN_CHECKS as u64 == 0 { check()?; }
        let subset = s ^ (s >> 1);
        let j = (subset ^ previous).trailing_zeros() as usize;
        let added = subset >> j & 1 == 1;
        for (i, sum) in sums.iter_mut().enumerate() { if added { *sum += &a[i * n + j] } else { *sum -= &a[i * n + j] } }
        previous = subset;
        let product = sums.iter().skip(1).fold(sums[0].clone(), |mut p, v| { p *= v; p });
        if (n - subset.count_ones() as usize) % 2 == 0 { total += product } else { total -= product }
    }
    Ok(if n == 0 { T::one() } else { total })
}

/// [`permanent_with`] with no check.
pub fn permanent<T: Clone + NumAssignRef>(a: &[T], n: usize) -> T { let Ok(p) = permanent_with(a, n, ok); p }
