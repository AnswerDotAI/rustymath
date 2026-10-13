//! The symmetric and Hermitian eigen problems, as LAPACK's `dsytrd`/`zhetrd`, `dorgtr`/`zungtr`, `dsteqr` and `dstedc` solve them.
//! Householder reflections reduce the matrix to a real symmetric tridiagonal one. Up to `CONQUER` rows, implicit QL iterations with
//! Wilkinson shifts diagonalise the tridiagonal matrix, rotating the rows of Qᵀ into the rows of Vᵀ. Larger tridiagonal matrices go to
//! divide and conquer, which builds their eigenvectors through matrix products. The reflections then turn those eigenvectors into V. Eigenvalues alone take implicit QL without rotations, at any size.
//! Every reflection and rotation acts on contiguous rows. The reduction works on the upper triangle. A Hermitian matrix's reduction
//! conjugates wherever it reads a row of the upper triangle as the column below the diagonal. Either solver then runs unchanged on
//! the real tridiagonal matrix that the reduction leaves.
//! Divide and conquer's deflation and merge follow faer's `divide_and_conquer_recurse` in `tridiag_evd.rs`. See `THIRD-PARTY.md`.
use super::*;

/// The eigen decomposition of a Hermitian `n`×`n` matrix: `a = V diag(values) Vᴴ`. `values` is real and in ascending order. Column
/// `j` of the `n`×`n` matrix `vectors` is a unit eigenvector for `values[j]`. Without eigenvectors, `vectors` is empty.
#[derive(Clone, Debug, PartialEq)]
pub struct Eigh<T: Scalar> {
    pub values: Vec<T::Real>,
    pub vectors: Vec<T>,
}

/// The eigen decomposition of the Hermitian `n`×`n` matrix `a`, reading only its lower triangle. `None` where the iterations don't
/// converge, as for a matrix holding NaN or an infinity.
pub fn eigh<T: Factor>(a: &[T], n: usize) -> Option<Eigh<T>> { T::matrix_eigh(a, n, true) }
/// The eigenvalues of the Hermitian `n`×`n` matrix `a`, in ascending order, reading only its lower triangle. `None` where `eigh` gives
/// `None`.
pub fn hermitian_eigenvalues<T: Factor>(a: &[T], n: usize) -> Option<Vec<T::Real>> { T::matrix_eigh(a, n, false).map(|f| f.values) }

/// `eigh` and `hermitian_eigenvalues` for `T`, with eigenvectors only where `vectors` is true. A matrix whose largest item lies outside
/// the safe range first scales by a power of two. Its eigenvalues then scale back exactly.
pub(super) fn eigh_factor<T: Scalar + Tiled>(a: &[T], n: usize, vectors: bool) -> Option<Eigh<T>>
where
    T::Real: Tiled,
{
    assert_eq!(a.len(), n * n, "`a` must hold `n`×`n` items");
    // Row i of `w` holds the conjugate of column i of `a`'s lower triangle, from the diagonal on: row i of the matrix's upper triangle.
    // The rest of `w` is zero.
    let mut w = adjoint(a, n, n);
    for i in 0..n { for x in &mut w[i * n..i * n + i] { *x = T::zero() } }
    let scale = safe_scale(&w);
    if let Some(s) = scale { for x in &mut w { *x = x.mul_real(s) } }
    let (mut d, mut e, taus) = tridiagonal(&mut w, n);
    let vt = if !vectors {
        // Values alone need no rotations, so QL takes O(n²) time after the reduction, at any size.
        implicit_ql::<T>(&mut d, &mut e, &mut [], n)?;
        Vec::new()
    } else if n > CONQUER {
        // Divide and conquer gives the tridiagonal matrix's eigenvectors as the rows of Sᵀ. The reflections then turn S into V = Q S
        // in place, acting on its rows 1 to n - 1.
        let st = conquer(&mut d, &e[..n - 1], CONQUER)?;
        let s: Vec<T> = st.into_iter().map(T::from_real).collect();
        let mut v = transpose(&s, n, n);
        blocked(&w, &taus, n).apply(&mut v[n..], n);
        transpose(&v, n, n)
    } else {
        let mut vt = reflections(&w, &taus, n);
        implicit_ql(&mut d, &mut e, &mut vt, n)?;
        vt
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap_or(Ordering::Equal));
    let values = order.iter().map(|&i| scale.map_or(d[i], |s| d[i] / s)).collect();
    let vectors = if vt.is_empty() { Vec::new() } else { (0..n * n).map(|ij| vt[order[ij % n] * n + ij / n]).collect() };
    Some(Eigh { values, vectors })
}

/// Reduces the Hermitian matrix whose upper triangle `w` holds, row by row, to a real symmetric tridiagonal one, and gives its diagonal,
/// its subdiagonal and each reflection's `τ`. Reflection `k` maps the column below row `k`'s diagonal, the conjugate of the row's items
/// after it, to a real multiple of its first unit vector. Its vector is 1 at column `k + 1`, and `w` keeps the rest of it in row `k`, from
/// column `k + 2`. Each reflection then updates the upper triangle below its row from both sides, as a rank-two update. That update needs
/// `τ B v`, where B is the matrix below and to the right of row `k`. One pass over B's rows applies reflection `k`'s update and accumulates
/// the next reflection's `τ B v`. B is then read once for each reflection.
fn tridiagonal<T: Scalar>(w: &mut [T], n: usize) -> (Vec<T::Real>, Vec<T::Real>, Vec<T>) {
    let zero = T::Real::zero();
    let (mut d, mut e, mut taus) = (vec![zero; n], vec![zero; n], vec![T::zero(); n.saturating_sub(1)]);
    if n == 0 { return (d, e, taus) }
    let (mut v, mut y, mut v2, mut y2) = (vec![T::zero(); n], vec![T::zero(); n], vec![T::zero(); n], vec![T::zero(); n]);
    if n > 1 {
        // Reflection 0 takes a pass of its own for its τ B v.
        let (head, rest) = w.split_at_mut(n);
        taus[0] = householder_row(&mut head[1..], &mut e[0], &mut v[..n - 1]);
        if taus[0] != T::zero() {
            for i in 0..n - 1 { symmetric_row(&mut y[..n - 1], &v[..n - 1], &rest[i * n + 1 + i..(i + 1) * n], i) }
            for t in &mut y[..n - 1] { *t = taus[0] * *t }
        }
    }
    for k in 0..n - 1 {
        d[k] = w[k * n + k].re();
        let (m, tau) = (n - k - 1, taus[k]);
        let rest = &mut w[(k + 1) * n..];
        // B ← B - v uᴴ - u vᴴ, where u = y - (τ/2)(yᴴv) v. B's first row updates first, because the next reflection comes from it.
        if tau != T::zero() {
            let alpha = -(tau * dotc(&y[..m], &v[..m])).div_real(T::Real::one() + T::Real::one());
            axpy(&mut y[..m], alpha, &v[..m]);
            let row = &mut rest[k + 1..n];
            axpyc(row, -v[0], &y[..m]);
            axpyc(row, -y[0], &v[..m]);
        }
        let tau2 = if m > 1 { householder_row(&mut rest[k + 2..n], &mut e[k + 1], &mut v2[..m - 1]) } else { T::zero() };
        if m > 1 { taus[k + 1] = tau2 }
        if tau2 != T::zero() { y2[..m - 1].fill(T::zero()) }
        for i in 1..m {
            let row = &mut rest[i * n + k + 1 + i..(i + 1) * n];
            if tau != T::zero() {
                axpyc(row, -v[i], &y[i..m]);
                axpyc(row, -y[i], &v[i..m]);
            }
            // B's row i is row i - 1 of the next reflection's B.
            if tau2 != T::zero() { symmetric_row(&mut y2[..m - 1], &v2[..m - 1], row, i - 1) }
        }
        if tau2 != T::zero() { for t in &mut y2[..m - 1] { *t = tau2 * *t } }
        std::mem::swap(&mut v, &mut v2);
        std::mem::swap(&mut y, &mut y2);
    }
    d[n - 1] = w[n * n - 1].re();
    (d, e, taus)
}

/// Turns `x`, a row's items after its diagonal, into a reflection of their conjugates, the column below the diagonal, as `reflector`
/// does, and gives its `τ`. `e` takes the subdiagonal item, and `v` the reflection's vector, with its leading 1.
fn householder_row<T: Scalar>(x: &mut [T], e: &mut T::Real, v: &mut [T]) -> T {
    for z in x.iter_mut() { *z = z.conj() }
    let tau = reflector(x);
    *e = x[0].re();
    v[0] = T::one();
    v[1..].copy_from_slice(&x[1..]);
    tau
}

/// Adds row `i` of a Hermitian matrix B to `y = B v`, where `row` holds the row's items from the diagonal on. Their conjugates after
/// the diagonal also stand for column `i` below the diagonal.
#[inline(always)]
fn symmetric_row<T: Scalar>(y: &mut [T], v: &[T], row: &[T], i: usize) {
    y[i] = y[i] + dot(row, &v[i..]);
    axpyc(&mut y[i + 1..], v[i], &row[1..]);
}

/// Rows at or below which `reflections` applies each reflection on its own, where building the WY form's blocks costs more than
/// its products save.
const SMALL: usize = 96;

/// Qᵀ, the transpose without conjugation, for the reflections that `tridiagonal` leaves in `w` and `taus`. Reflection `k` acts on rows
/// and columns from `k + 1` on, as reflection `k` of a QR factorization acts on a matrix without the first row and column.
/// `Reflections::q` therefore forms Q's last `n - 1` rows and columns, by blocks of reflections in the WY form, through the tiled `+.×`
/// kernels.
pub(super) fn reflections<T: Scalar + Tiled>(w: &[T], taus: &[T], n: usize) -> Vec<T> {
    let mut qt = identity(n);
    if n < 2 { return qt }
    if n <= SMALL {
        // A small matrix applies each reflection's transpose, `I - τ v̄ vᵀ`, from the right on its own. Before reflection `k`, Qᵀ
        // differs from the identity only in rows and columns after `k + 1`.
        let mut v = vec![T::zero(); n];
        for (k, &tau) in taus.iter().enumerate().rev() {
            if tau == T::zero() { continue }
            let v = &mut v[..n - k - 1];
            v[0] = T::one();
            v[1..].copy_from_slice(&w[k * n + k + 2..(k + 1) * n]);
            for row in qt[(k + 1) * n..].chunks_exact_mut(n) {
                let row = &mut row[k + 1..];
                let s = tau * dotc(v, row);
                axpy(row, -s, v);
            }
        }
        return qt;
    }
    let p = n - 1;
    transpose_into(&blocked(w, taus, n).q(), p, &mut qt[n + 1..], n, [p, p], |x| x);
    qt
}

/// The reflections that `reflections` multiplies, in blocks, for a matrix of at least 2 rows. They act on rows and columns 1 to `n - 1`.
pub(super) fn blocked<T: Scalar + Tiled>(w: &[T], taus: &[T], n: usize) -> Reflections<T> {
    let (p, width) = (n - 1, qr_block(n - 1));
    let blocks = (0..p)
        .step_by(width)
        .map(|k0| {
            let nb = (k0 + width).min(p) - k0;
            // Row c of the panel holds reflection k0 + c from column k0 + 1 of `w`, where `householder` reads it.
            let panel = (0..nb).flat_map(|c| w[(k0 + c) * n + k0 + 1..(k0 + c + 1) * n].to_vec()).collect();
            Block::new(k0, panel, &taus[k0..k0 + nb])
        })
        .collect();
    Reflections { m: p, blocks }
}

/// Diagonalises the real symmetric tridiagonal matrix with diagonal `d` and subdiagonal `e` by implicit QL iterations, as EISPACK's
/// `tql2` does. Each rotation also rotates two adjacent rows of `vt`, which starts as Qᵀ and ends as Vᵀ, unless `vt` is empty. `d`
/// ends as the eigenvalues, in no particular order. `None` where an eigenvalue takes more than 30 iterations.
fn implicit_ql<T: Scalar>(d: &mut [T::Real], e: &mut [T::Real], vt: &mut [T], n: usize) -> Option<()> {
    let (zero, one) = (T::Real::zero(), T::Real::one());
    let hypot = |x: T::Real, y: T::Real| x.mla::<FUSED>(x, y * y).sqrt();
    for l in 0..n {
        let mut iterations = 0;
        loop {
            // The first negligible subdiagonal item at or after l splits off the block that holds d[l].
            let mut m = l;
            // A NaN never counts as negligible, so a matrix holding one runs out of iterations.
            while m + 1 < n && !(e[m].abs() <= T::Real::epsilon() * (d[m].abs() + d[m + 1].abs())) { m += 1 }
            if m == l { break }
            iterations += 1;
            if iterations > 30 { return None }
            // The Wilkinson shift: the eigenvalue of the leading 2×2 block nearer d[l].
            let g = (d[l + 1] - d[l]) / (e[l] + e[l]);
            let r = hypot(g, one);
            let mut g = d[m] - d[l] + e[l] / (g + if g < zero { -r } else { r });
            let (mut s, mut c, mut p) = (one, one, zero);
            let mut split = false;
            for i in (l..m).rev() {
                let (f, b) = (s * e[i], c * e[i]);
                let r = hypot(f, g);
                e[i + 1] = r;
                if r == zero {
                    // An exact zero splits the block: d[i + 1] has converged.
                    d[i + 1] = d[i + 1] - p;
                    e[m] = zero;
                    split = true;
                    break;
                }
                (s, c) = (f / r, g / r);
                g = d[i + 1] - p;
                let r = (d[i] - g) * s + (c + c) * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                if !vt.is_empty() {
                    let (x, y) = two_rows(vt, n, i, i + 1);
                    rotate(x, y, c, s);
                }
            }
            if split { continue }
            d[l] = d[l] - p;
            e[l] = g;
            e[m] = zero;
        }
    }
    Some(())
}


/// Above this many rows, divide and conquer splits a tridiagonal matrix's eigen problem in two. Implicit QL solves smaller ones.
const CONQUER: usize = 64;

/// The eigen decomposition of the real symmetric tridiagonal matrix with diagonal `d` and subdiagonal `e`, by Cuppen's divide and
/// conquer with Gu and Eisenstat's eigenvectors, as LAPACK's `dstedc` computes it. `d` ends
/// holding the eigenvalues in ascending order. The rows of the `n`×`n` result are unit eigenvectors for them. Problems of at most `leaf`
/// rows go to implicit QL. A matrix holding NaN or an infinity gives `None`.
fn conquer<R: Native + Tiled>(d: &mut [R], e: &[R], leaf: usize) -> Option<Vec<R>> {
    let n = d.len();
    if n <= leaf.max(1) {
        let mut st = identity(n);
        let mut e = [e, &[R::zero()]].concat();
        implicit_ql::<R>(d, &mut e, &mut st, n)?;
        return Some(ascending(d, &st));
    }
    let (n1, two) = (n / 2, R::one() + R::one());
    let rho = e[n1 - 1];
    if !rho.is_finite() { return None }
    // Tearing ρ out at the split leaves diag(T₀, T₁) + |ρ| w wᵀ, where w is 1 at rows n1 - 1 and n1, signed as ρ there.
    d[n1 - 1] = d[n1 - 1] - rho.abs();
    d[n1] = d[n1] - rho.abs();
    let (d0, d1) = d.split_at_mut(n1);
    let (s0, s1) = (conquer(d0, &e[..n1 - 1], leaf)?, conquer(d1, &e[n1..], leaf)?);
    // In the halves' eigenvectors, w/√2 is z: the last components of T₀'s vectors and the first of T₁'s. The update is then 2|ρ| z zᵀ.
    let n2 = n - n1;
    let r = two.sqrt().recip();
    let signed = if rho < R::zero() { -r } else { r };
    let z: Vec<R> = (0..n1).map(|k| s0[k * n1 + n1 - 1] * r).chain((0..n2).map(|k| s1[k * n2] * signed)).collect();
    Some(merge(d, &z, rho.abs() * two, &s0, &s1, n1))
}

/// `d` in ascending order, with the rows of the `n`×`n` matrix `st` in the same order.
fn ascending<R: Native>(d: &mut [R], st: &[R]) -> Vec<R> {
    let n = d.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap_or(Ordering::Equal));
    let sorted: Vec<R> = order.iter().map(|&i| d[i]).collect();
    d.copy_from_slice(&sorted);
    order.iter().flat_map(|&i| st[i * n..(i + 1) * n].iter().copied()).collect()
}

/// One eigenvalue of a merge: a root of the secular equation, by its index among the kept items, or a deflated item, by its position.
enum Source { Root(usize), Deflated(usize) }

/// The eigen decomposition of `diag(d) + rho z zᵀ`, turned into the whole tridiagonal matrix's. The first `n1` items of `d` are the
/// first half's eigenvalues, and the rest the second half's, each in ascending order. The rows of `s0` and `s1` hold the halves' unit
/// eigenvectors. Gives the whole matrix's unit eigenvectors as rows, and leaves its eigenvalues in ascending order in `d`.
fn merge<R: Native + Tiled>(d: &mut [R], z: &[R], rho: R, s0: &[R], s1: &[R], n1: usize) -> Vec<R> {
    let (n, zero, one) = (d.len(), R::zero(), R::one());
    let n2 = n - n1;
    // `p` lists the positions of `d` in ascending order of their values.
    let (mut p, mut i, mut j) = (Vec::with_capacity(n), 0, n1);
    while p.len() < n {
        if j == n || (i < n1 && d[i] < d[j]) {
            p.push(i);
            i += 1;
        } else {
            p.push(j);
            j += 1;
        }
    }
    let mut pd: Vec<R> = p.iter().map(|&k| d[k]).collect();
    let mut pz: Vec<R> = p.iter().map(|&k| z[k]).collect();
    let largest = |x: &[R]| x.iter().fold(zero, |m, v| m.max(v.abs()));
    let zmax = largest(&pz);
    let tol = R::narrow(8.0) * R::epsilon() * largest(&pd).max(zmax);
    let mut st = vec![zero; n * n];
    if rho * zmax <= tol {
        // The update is negligible, so the halves' eigenvectors are the whole matrix's.
        for (j, &k) in p.iter().enumerate() {
            if k < n1 {
                st[j * n..j * n + n1].copy_from_slice(&s0[k * n1..(k + 1) * n1]);
            } else {
                st[j * n + n1..(j + 1) * n].copy_from_slice(&s1[(k - n1) * n2..(k - n1 + 1) * n2]);
            }
        }
        d.copy_from_slice(&pd);
        return st;
    }
    // Deflation. A negligible item of z leaves its eigenvalue and vector unchanged. A run of eigenvalues within `tol` of each other
    // shares one eigenspace. A reflection within that space moves the run's part of z into its last item, deflating the others.
    for v in &mut pz { if (rho * *v).abs() <= tol { *v = zero } }
    let mut runs = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && pd[end] - pd[start] <= tol {
            pd[end] = pd[start];
            end += 1;
        }
        let x = &mut pz[start..end];
        let norm = x.iter().fold(zero, |s, &v| v.mla::<FUSED>(v, s)).sqrt();
        if x.len() > 1 && norm != zero {
            let last = x.len() - 1;
            let beta = if x[last] < zero { norm } else { -norm };
            let mut v = x.to_vec();
            v[last] = v[last] - beta;
            x.fill(zero);
            x[last] = beta;
            runs.push((start, v));
        }
        start = end;
    }
    let kept: Vec<usize> = (0..n).filter(|&i| pz[i] != zero).collect();
    let (dk, mut zk): (Vec<R>, Vec<R>) = kept.iter().map(|&i| (pd[i], pz[i])).unzip();
    let k = kept.len();
    // Each kept item's eigenvalue is a root of the secular equation 1/ρ + Σ zᵢ²/(dᵢ - λ) = 0, between consecutive dᵢ, or above the last.
    let inverse = rho.recip();
    let f = |shift: R, mu: R| dk.iter().zip(&zk).fold(inverse, |s, (&di, &zi)| s + zi * (zi / ((di - shift) - mu)));
    let four = |shift: [R; 4], mu: [R; 4]| {
        let mut s = [inverse; 4];
        for (&di, &zi) in dk.iter().zip(&zk) {
            for t in 0..4 { s[t] = s[t] + zi * (zi / ((di - shift[t]) - mu[t])) }
        }
        s
    };
    let norm2 = zk.iter().fold(zero, |s, &v| v.mla::<FUSED>(v, s));
    let roots: Vec<(R, R)> = (0..k)
        .map(|t| {
            let last = t + 1 == k;
            secular::root(&f, &four, dk[t], if last { dk[t] + rho * norm2 } else { dk[t + 1] }, last)
        })
        .collect();
    // Gu and Eisenstat's z: the vector for which the computed roots are exact eigenvalues. Its eigenvectors come out orthogonal.
    for t in 0..k {
        let di = dk[t];
        let mut product = roots[t].1 + (roots[t].0 - di);
        for s in (0..k).filter(|&s| s != t) { product = product * ((roots[s].1 + (roots[s].0 - di)) / (dk[s] - di)) }
        zk[t] = if zk[t] < zero { -product.abs().sqrt() } else { product.abs().sqrt() };
    }
    let mut values: Vec<(R, Source)> = roots.iter().enumerate().map(|(t, &(shift, mu))| (shift + mu, Source::Root(t))).collect();
    values.extend((0..n).filter(|&i| pz[i] == zero).map(|i| (pd[i], Source::Deflated(i))));
    values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    // The rows of `rt` are the update's unit eigenvectors, with their components in the order of `p`.
    let mut rt = vec![zero; n * n];
    for (row, (_, source)) in rt.chunks_exact_mut(n).zip(&values) {
        match *source {
            Source::Deflated(i) => row[i] = one,
            Source::Root(t) => {
                let (shift, mu) = roots[t];
                for (s, &i) in kept.iter().enumerate() { row[i] = zk[s] / ((dk[s] - shift) - mu) }
                let norm = row.iter().fold(zero, |s, &v| v.mla::<FUSED>(v, s)).sqrt();
                for v in row.iter_mut() { *v = *v / norm }
            }
        }
    }
    for (start, v) in &runs {
        let c = (one + one) / dot(v, v);
        for row in rt.chunks_exact_mut(n) {
            let segment = &mut row[*start..*start + v.len()];
            let s = c * dot(v, segment);
            axpy(segment, -s, v);
        }
    }
    for (x, &(value, _)) in d.iter_mut().zip(&values) { *x = value }
    // The whole matrix's eigenvectors: each row of `rt`, with its components back in the halves' order, times diag(S₀ᵀ, S₁ᵀ).
    let mut q = vec![0; n];
    for (i, &k) in p.iter().enumerate() { q[k] = i }
    R::add_product(&mut st, n, |j, c| rt[j * n + q[c]], |c, x| s0[c * n1 + x], [n, n1, n1]);
    R::add_product(&mut st[n1..], n, |j, c| rt[j * n + q[n1 + c]], |c, x| s1[c * n2 + x], [n, n2, n2]);
    st
}

#[cfg(test)]
mod tests {
    use super::super::tests::{distance, norm, orthogonality, product, random, RealItem, wide};
    use super::*;

    /// The symmetric `n`×`n` matrix whose lower triangle comes from `lower`, with `junk` in its upper triangle, which `eigh` never reads.
    fn symmetric<T: RealItem>(lower: &[T], n: usize, junk: T) -> Vec<T> {
        (0..n * n).map(|k| if k % n > k / n { junk } else { lower[k] }).collect()
    }

    /// The symmetric tridiagonal `n`×`n` matrix with diagonal `d` and subdiagonal `e`, in full.
    fn tridiagonal_matrix(d: &[f64], e: &[f64]) -> Vec<f64> {
        let n = d.len();
        (0..n * n)
            .map(|k| match (k / n, k % n) {
                (i, j) if i == j => d[i],
                (i, j) if i == j + 1 || j == i + 1 => e[i.min(j)],
                _ => 0.0,
            })
            .collect()
    }

    /// Checks that `f` decomposes the symmetric `n`×`n` matrix `a`: ascending values, orthonormal vectors, and a small residual
    /// `A V - V Λ`, all within `bound`.
    fn check(a: &[f64], n: usize, f: &Eigh<f64>, bound: f64, what: &str) {
        assert!(f.values.windows(2).all(|w| w[0] <= w[1]), "values ascend for {what}");
        let (v, scale) = (&f.vectors, norm(a).max(f64::MIN_POSITIVE));
        let vl: Vec<f64> = (0..n * n).map(|k| v[k] * f.values[k % n]).collect();
        assert!(distance(&product(a, v, [n; 3]), &vl, scale) < bound, "residual for {what}");
        assert!(orthogonality(v, n, n) < bound, "orthogonal vectors for {what}");
    }

    /// Checks `eigh` at `T` on random symmetric matrices, where `tolerance` is a few times `T`'s epsilon.
    fn random_matrices<T: RealItem>(tolerance: f64) {
        for n in [0, 1, 2, 5, 31, 64, 129] {
            let lower = random::<T>(n, n, n as u64 + 3);
            let a = symmetric(&lower, n, T::zero());
            let full: Vec<T> = (0..n * n).map(|k| lower[if k % n > k / n { k % n * n + k / n } else { k }]).collect();
            let f = eigh(&a, n).expect("a symmetric matrix converges");
            assert_eq!(eigh(&symmetric(&lower, n, T::narrow(1e30)), n).as_ref(), Some(&f), "eigh reads only the lower triangle at {n}");
            let g = Eigh { values: wide(&f.values), vectors: wide(&f.vectors) };
            check(&wide(&full), n, &g, tolerance * (n as f64).max(1.0), &format!("{n}×{n}"));
        }
    }

    #[test]
    fn symmetric_eigen_has_small_residuals() {
        random_matrices::<f64>(4.0 * f64::EPSILON);
        random_matrices::<f32>(4.0 * f32::EPSILON as f64);
    }

    #[test]
    fn symmetric_eigen_survives_hard_spectra() {
        let bound = 64.0 * f64::EPSILON;
        // Repeated eigenvalues: the identity, a zero matrix and a diagonal matrix with ties.
        for (a, n, values) in [(identity::<f64>(4), 4, vec![1.0; 4]), (vec![0.0; 9], 3, vec![0.0; 3]), (tridiagonal_matrix(&[2.0, 1.0, 2.0, 1.0], &[0.0; 3]), 4, vec![1.0, 1.0, 2.0, 2.0])] {
            let f = eigh(&a, n).unwrap();
            assert_eq!(f.values, values, "eigenvalues of a matrix with ties");
            check(&a, n, &f, bound, "a matrix with ties");
        }
        // Wilkinson's matrix W7⁺, whose largest eigenvalues come in close pairs, and two copies of it joined weakly.
        let wilkinson = [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let glued_d: Vec<f64> = wilkinson.iter().chain(&wilkinson).copied().collect();
        let glued_e: Vec<f64> = (0..13).map(|i| if i == 6 { 1e-6 } else { 1.0 }).collect();
        for (d, e) in [(wilkinson.to_vec(), vec![1.0; 6]), (glued_d, glued_e)] {
            let (a, n) = (tridiagonal_matrix(&d, &e), d.len());
            check(&a, n, &eigh(&a, n).unwrap(), bound, "a Wilkinson matrix");
        }
        // A 2×2 block whose subdiagonal is barely above the negligible size.
        let a = tridiagonal_matrix(&[1.0000000000000002, 1.0000000000000002], &[7.216449660063518e-16]);
        check(&a, 2, &eigh(&a, 2).unwrap(), bound, "a nearly diagonal 2×2 matrix");
    }

    #[test]
    fn divide_and_conquer_survives_hard_spectra() {
        // Leaves of 2 and 4 rows make every merge path run: negligible updates, zero items of z, runs of ties and secular roots.
        let wilkinson = [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let glued_d: Vec<f64> = wilkinson.iter().chain(&wilkinson).copied().collect();
        let glued_e: Vec<f64> = (0..13).map(|i| if i == 6 { 1e-6 } else { 1.0 }).collect();
        let random_d: Vec<f64> = random::<f64>(1, 100, 7);
        let random_e: Vec<f64> = random::<f64>(1, 99, 8);
        let clustered_d: Vec<f64> = (0..60).map(|i| if i < 30 { 1.0 } else { 2.0 }).collect();
        let clustered_e: Vec<f64> = (0..59).map(|i| if i % 7 == 0 { 0.5 } else { 1e-18 }).collect();
        let cases = [
            (wilkinson.to_vec(), vec![1.0; 6], "a Wilkinson matrix"),
            (glued_d, glued_e, "two glued Wilkinson matrices"),
            (random_d, random_e, "a random tridiagonal matrix"),
            (clustered_d, clustered_e, "a matrix with two clusters of eigenvalues"),
            (vec![1.0; 50], vec![0.0; 49], "the identity"),
            ((0..40).map(|i| (i % 2 + 1) as f64).collect(), vec![0.0; 39], "a diagonal matrix with ties"),
        ];
        for (d, e, what) in cases {
            let (a, n) = (tridiagonal_matrix(&d, &e), d.len());
            for leaf in [2, 4] {
                let mut values = d.clone();
                let st = conquer(&mut values, &e, leaf).expect("a tridiagonal matrix converges");
                assert!(values.windows(2).all(|w| w[0] <= w[1]), "values ascend for {what}");
                let v = transpose(&st, n, n);
                let vl: Vec<f64> = (0..n * n).map(|k| v[k] * values[k % n]).collect();
                let bound = 64.0 * f64::EPSILON * n as f64;
                assert!(distance(&product(&a, &v, [n; 3]), &vl, norm(&a).max(f64::MIN_POSITIVE)) < bound, "residual for {what}");
                assert!(orthogonality(&v, n, n) < bound, "orthogonal vectors for {what}");
            }
        }
    }

    /// Checks that `eigh` at `T` scales exactly with a matrix scaled by a power of two near underflow or overflow, and gives `None` for
    /// a matrix holding NaN.
    fn hard_matrices<T: RealItem>() {
        let n = 40;
        let a = symmetric(&random::<T>(n, n, 41), n, T::zero());
        let f = eigh(&a, n).unwrap();
        let tiny = T::min_positive_value() / T::epsilon();
        for s in [tiny, tiny.recip()] {
            let g = eigh(&a.iter().map(|&v| v * s).collect::<Vec<T>>(), n).unwrap();
            assert_eq!(g.vectors, f.vectors, "eigenvectors scaled by {s:?}");
            assert_eq!(g.values, f.values.iter().map(|&v| v * s).collect::<Vec<T>>(), "eigenvalues scaled by {s:?}");
        }
        let mut nan = a.clone();
        nan[3 * n + 1] = T::nan();
        assert!(eigh(&nan, n).is_none(), "a matrix holding NaN");
    }

    #[test]
    fn symmetric_eigen_scales_exactly() {
        hard_matrices::<f64>();
        hard_matrices::<f32>();
    }
}
