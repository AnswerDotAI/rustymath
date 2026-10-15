//! The symmetric and Hermitian eigen problems, as LAPACK's `dsytrd`/`zhetrd`, `dorgtr`/`zungtr`, `dsteqr` and `dstedc` solve them. Householder reflections reduce the matrix to a real symmetric tridiagonal one. Up to `EIGH_CONQUER` rows, implicit QL iterations with Wilkinson shifts diagonalise the tridiagonal matrix, rotating the rows of Qᵀ into the rows of Vᵀ. Larger tridiagonal matrices go to divide and conquer, which builds their eigenvectors through matrix products. The reflections then turn those eigenvectors into V. Eigenvalues alone take implicit QL without rotations, at any size. Every reflection and rotation acts on contiguous rows. The reduction works on the upper triangle. A Hermitian matrix's reduction conjugates wherever it reads a row of the upper triangle as the column below the diagonal. Either solver then runs unchanged on the real tridiagonal matrix that the reduction leaves. Divide and conquer's deflation and merge follow faer's `divide_and_conquer_recurse` in `tridiag_evd.rs`. See `THIRD-PARTY.md`.
use super::*;

/// The native eigen decomposition of the nonempty Hermitian `n`×`n` matrix whose finite lower triangle `a` holds, which `Tiled::eigh` uses off macOS. A matrix whose largest item lies outside the safe range first scales by a power of two. Its eigenvalues then scale back exactly.
pub(in crate::linalg) fn native<T: Tiled>(a: &[T], n: usize, vectors: bool) -> Option<Eigh<T>> where T::Real: Tiled<Real = T::Real> {
    // Row i of `w` holds the conjugate of column i of `a`'s lower triangle, from the diagonal on: row i of the matrix's upper triangle. The rest of `w` is zero.
    let mut w = adjoint(a, n, n);
    for i in 0..n { for x in &mut w[i * n..i * n + i] { *x = T::zero() } }
    let scale = safe_scale(&w);
    if let Some(s) = scale { for x in &mut w { *x = x.mul_real(s) } }
    let (mut d, mut e, taus) = tridiagonal(&mut w, n);
    let vt = if !vectors {
        // Values alone need no rotations, so QL takes O(n²) time after the reduction, at any size.
        implicit_ql::<T>(&mut d, &mut e, &mut [], n)?;
        Vec::new()
    } else if n > EIGH_CONQUER {
        // Divide and conquer gives the tridiagonal matrix's eigenvectors as the rows of Sᵀ. The reflections then turn S into V = Q S in place, acting on its rows 1 to n - 1.
        let st = conquer(&mut d, &e[..n - 1], EIGH_CONQUER)?;
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

/// Reduces the Hermitian matrix whose upper triangle `w` holds, row by row, to a real symmetric tridiagonal one, and gives its diagonal, its subdiagonal and each reflection's `τ`. Reflection `k` maps the column below row `k`'s diagonal, the conjugate of the row's items after it, to a real multiple of its first unit vector. Its vector is 1 at column `k + 1`, and `w` keeps the rest of it in row `k`, from column `k + 2`. Each reflection then updates the upper triangle below its row from both sides, as a rank-two update. That update needs `τ B v`, where B is the matrix below and to the right of row `k`. One pass over B's rows applies reflection `k`'s update and accumulates the next reflection's `τ B v`. B is then read once for each reflection. `summed_rows` splits each pass's rows across threads.
fn tridiagonal<T: Tiled>(w: &mut [T], n: usize) -> (Vec<T::Real>, Vec<T::Real>, Vec<T>) {
    let zero = T::Real::zero();
    let (mut d, mut e, mut taus) = (vec![zero; n], vec![zero; n], vec![T::zero(); n.saturating_sub(1)]);
    if n == 0 { return (d, e, taus); }
    let (mut v, mut y, mut v2, mut y2) = (vec![T::zero(); n], vec![T::zero(); n], vec![T::zero(); n], vec![T::zero(); n]);
    let mut parts = Vec::new();
    if n > 1 {
        // Reflection 0 takes a pass of its own for its τ B v.
        let (head, rest) = w.split_at_mut(n);
        taus[0] = householder_row(&mut head[1..], &mut e[0], &mut v[..n - 1]);
        if taus[0] != T::zero() {
            summed_rows(&mut rest[..(n - 1) * n], n, n - 1, &mut y[..n - 1], &mut parts, |i, items, sums| symmetric_row(sums, &v[..n - 1], &items[1 + i..], i));
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
        // B's row i + 1 is row i of the next reflection's B.
        summed_rows(&mut rest[n..m * n], n, m - 1, &mut y2[..m - 1], &mut parts, |i, items, sums| {
            let row = &mut items[k + 2 + i..];
            if tau != T::zero() {
                axpyc(row, -v[i + 1], &y[i + 1..m]);
                axpyc(row, -y[i + 1], &v[i + 1..m]);
            }
            if tau2 != T::zero() { symmetric_row(sums, &v2[..m - 1], row, i) }
        });
        if tau2 != T::zero() { for t in &mut y2[..m - 1] { *t = tau2 * *t } }
        std::mem::swap(&mut v, &mut v2);
        std::mem::swap(&mut y, &mut y2);
    }
    d[n - 1] = w[n * n - 1].re();
    (d, e, taus)
}

/// Turns `x`, a row's items after its diagonal, into a reflection of their conjugates, the column below the diagonal, as `reflector` does, and gives its `τ`. `e` takes the subdiagonal item, and `v` the reflection's vector, with its leading 1.
fn householder_row<T: Scalar>(x: &mut [T], e: &mut T::Real, v: &mut [T]) -> T {
    for z in x.iter_mut() { *z = z.conj() }
    let tau = reflector(x);
    *e = x[0].re();
    v[0] = T::one();
    v[1..].copy_from_slice(&x[1..]);
    tau
}

/// Adds row `i` of a Hermitian matrix B to `y = B v`, where `row` holds the row's items from the diagonal on. Their conjugates after the diagonal also stand for column `i` below the diagonal.
#[inline(always)]
fn symmetric_row<T: Scalar>(y: &mut [T], v: &[T], row: &[T], i: usize) {
    y[i] = y[i] + dot(row, &v[i..]);
    axpyc(&mut y[i + 1..], v[i], &row[1..]);
}

/// The fewest rows in each piece of a reduction's pass. At 64, values-only SVD and eigen at 1024 ran 1.65 to 2 times faster on 18 threads than as one piece.
const ROW_PIECE: usize = 64;

/// Runs `row(i, items, sums)` on each of rows `0..count` of a reduction's pass, where row `i`'s items are `rows[i * stride..(i + 1) * stride]`, and adds the sums each row gives into `sums`. The rows split into one piece of at least `ROW_PIECE` rows for each of this thread's [`crate::parallel::threads`]. On one thread they run as one loop. The first piece adds into `sums` and each later one into its own part of `parts`, which then add into `sums`.
pub(super) fn summed_rows<T: Tiled, F: Fn(usize, &mut [T], &mut [T]) + Sync>(
    rows: &mut [T], stride: usize, count: usize, sums: &mut [T], parts: &mut Vec<T>, row: F) {
    let pieces = (count / ROW_PIECE).min(crate::parallel::threads());
    if pieces <= 1 {
        for (i, items) in rows.chunks_mut(stride).take(count).enumerate() { row(i, items, sums) }
        return;
    }
    let per = count.div_ceil(pieces);
    let (len, width) = (sums.len(), sums.len().max(1));
    parts.clear();
    parts.resize((count.div_ceil(per) - 1) * width, T::zero());
    {
        let (first, rest) = rows[..count * stride].split_at_mut(per * stride);
        let mut pieces: Vec<(&mut [T], &mut [T])> =
            std::iter::once((first, &mut *sums)).chain(rest.chunks_mut(per * stride).zip(parts.chunks_mut(width).map(|part| &mut part[..len]))).collect();
        crate::parallel::pieces_of(&mut pieces, 1, usize::MAX, &|p, piece| {
            let (rows, part) = &mut piece[0];
            for (j, items) in rows.chunks_mut(stride).enumerate() { row(p * per + j, items, part) }
            true
        });
    }
    for part in parts.chunks(width) { for (s, &x) in sums.iter_mut().zip(part) { *s = *s + x } }
}

/// Diagonalises the real symmetric tridiagonal matrix with diagonal `d` and subdiagonal `e` by implicit QL iterations, as EISPACK's `tql2` does. Each rotation also rotates two adjacent rows of `vt`, which starts as Qᵀ and ends as Vᵀ, unless `vt` is empty. `d` ends as the eigenvalues, in no particular order. `None` where an eigenvalue takes more than 30 iterations.
fn implicit_ql<T: Scalar>(d: &mut [T::Real], e: &mut [T::Real], vt: &mut [T], n: usize) -> Option<()> {
    let (zero, one) = (T::Real::zero(), T::Real::one());
    // The reduction to tridiagonal form makes errors as large as ε times the matrix's norm. Any subdiagonal item below that size is negligible. A test against the item's two neighbours alone never splits off the rounding noise that a low-rank matrix leaves.
    let norm = d.iter().zip(e.iter()).map(|(x, y)| x.abs() + y.abs()).fold(zero, |m, r| if r.is_nan() || r > m { r } else { m });
    let small = T::Real::epsilon() * norm;
    for l in 0..n {
        let mut iterations = 0;
        loop {
            // The first negligible subdiagonal item at or after l splits off the block that holds d[l].
            let mut m = l;
            // A NaN anywhere makes `small` NaN. A matrix holding one then runs out of iterations.
            while m + 1 < n && !(e[m].abs() <= small) { m += 1 }
            if m == l { break; }
            iterations += 1;
            if iterations > 30 { return None; }
            // The Wilkinson shift: the eigenvalue of the leading 2×2 block nearer d[l].
            let g = (d[l + 1] - d[l]) / (e[l] + e[l]);
            let r = plain_hypot(g, one);
            let mut g = d[m] - d[l] + e[l] / (g + if g < zero { -r } else { r });
            let (mut s, mut c, mut p) = (one, one, zero);
            let mut split = false;
            for i in (l..m).rev() {
                let (f, b) = (s * e[i], c * e[i]);
                let r;
                (c, s, r) = givens(g, f);
                e[i + 1] = r;
                if r == zero {
                    // An exact zero splits the block: d[i + 1] has converged.
                    d[i + 1] = d[i + 1] - p;
                    e[m] = zero;
                    split = true;
                    break;
                }
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
            if split { continue; }
            d[l] = d[l] - p;
            e[l] = g;
            e[m] = zero;
        }
    }
    Some(())
}

/// The eigen decomposition of the real symmetric tridiagonal matrix with diagonal `d` and subdiagonal `e`, by Cuppen's divide and conquer with Gu and Eisenstat's eigenvectors, as LAPACK's `dstedc` computes it. `d` ends holding the eigenvalues in ascending order. The rows of the `n`×`n` result are unit eigenvectors for them. Problems of at most `leaf` rows go to implicit QL. A matrix holding NaN or an infinity gives `None`.
fn conquer<R: Native + Tiled<Real = R>>(d: &mut [R], e: &[R], leaf: usize) -> Option<Vec<R>> {
    let n = d.len();
    if n <= leaf.max(1) {
        let mut st = identity(n);
        let mut e = [e, &[R::zero()]].concat();
        implicit_ql::<R>(d, &mut e, &mut st, n)?;
        return Some(ascending(d, &st));
    }
    let (n1, two) = (n / 2, R::one() + R::one());
    let rho = e[n1 - 1];
    if !rho.is_finite() { return None; }
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

/// The eigen decomposition of `diag(d) + rho z zᵀ`, turned into the whole tridiagonal matrix's. The first `n1` items of `d` are the first half's eigenvalues, and the rest the second half's, each in ascending order. The rows of `s0` and `s1` hold the halves' unit eigenvectors. Gives the whole matrix's unit eigenvectors as rows, and leaves its eigenvalues in ascending order in `d`.
fn merge<R: Native + Tiled<Real = R>>(d: &mut [R], z: &[R], rho: R, s0: &[R], s1: &[R], n1: usize) -> Vec<R> {
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
            if k < n1 { st[j * n..j * n + n1].copy_from_slice(&s0[k * n1..(k + 1) * n1]); }
            else { st[j * n + n1..(j + 1) * n].copy_from_slice(&s1[(k - n1) * n2..(k - n1 + 1) * n2]); }
        }
        d.copy_from_slice(&pd);
        return st;
    }
    // Deflation. A negligible item of z leaves its eigenvalue and vector unchanged. A run of eigenvalues within `tol` of each other shares one eigenspace. A reflection within that space moves the run's part of z into its last item, deflating the others.
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
            runs.push((start, (one + one) / dot(&v, &v), v));
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
        for (&di, &zi) in dk.iter().zip(&zk) { for t in 0..4 { s[t] = s[t] + zi * (zi / ((di - shift[t]) - mu[t])) } }
        s
    };
    let norm2 = zk.iter().fold(zero, |s, &v| v.mla::<FUSED>(v, s));
    let mut roots = vec![(zero, zero); k];
    crate::parallel::split(&mut roots, 1, &|first, out| {
        for (j, root) in out.iter_mut().enumerate() {
            let t = first + j;
            let last = t + 1 == k;
            *root = secular::root(&f, &four, dk[t], if last { dk[t] + rho * norm2 } else { dk[t + 1] }, last);
        }
        true
    });
    // Gu and Eisenstat's z: the vector for which the computed roots are exact eigenvalues. Its eigenvectors come out orthogonal.
    crate::parallel::split(&mut zk, 1, &|first, out| {
        for (j, z) in out.iter_mut().enumerate() {
            let t = first + j;
            let di = dk[t];
            let mut product = roots[t].1 + (roots[t].0 - di);
            for s in (0..k).filter(|&s| s != t) { product = product * ((roots[s].1 + (roots[s].0 - di)) / (dk[s] - di)) }
            *z = if *z < zero { -product.abs().sqrt() } else { product.abs().sqrt() };
        }
        true
    });
    let mut values: Vec<(R, Source)> = roots.iter().enumerate().map(|(t, &(shift, mu))| (shift + mu, Source::Root(t))).collect();
    values.extend((0..n).filter(|&i| pz[i] == zero).map(|i| (pd[i], Source::Deflated(i))));
    values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    // Build each update eigenvector in `p`'s order, then store it in the halves' original order for the products below.
    let mut rt = vec![zero; n * n];
    crate::parallel::split(&mut rt, n, &|first, rows| {
        let mut row = filled(n, zero);
        for (j, out) in rows.chunks_exact_mut(n).enumerate() {
            row.fill(zero);
            match values[first / n + j].1 {
                Source::Deflated(i) => row[i] = one,
                Source::Root(t) => {
                    let (shift, mu) = roots[t];
                    for (s, &i) in kept.iter().enumerate() { row[i] = zk[s] / ((dk[s] - shift) - mu) }
                    let norm = row.iter().fold(zero, |s, &v| v.mla::<FUSED>(v, s)).sqrt();
                    for v in row.iter_mut() { *v = *v / norm }
                }
            }
            for (start, c, v) in &runs {
                let segment = &mut row[*start..*start + v.len()];
                let s = *c * dot(v, segment);
                axpy(segment, -s, v);
            }
            for (i, &k) in p.iter().enumerate() { out[k] = row[i] }
        }
        true
    });
    for (x, &(value, _)) in d.iter_mut().zip(&values) { *x = value }
    // The whole matrix's eigenvectors: `rt` times diag(S₀ᵀ, S₁ᵀ).
    let rt = Matrix::rows(&rt, n);
    R::gemm(&mut st, n, rt, Matrix::rows(s0, n1), [n, n1, n1], one, zero);
    R::gemm(&mut st[n1..], n, rt.offset(0, n1), Matrix::rows(s1, n2), [n, n2, n2], one, zero);
    st
}
