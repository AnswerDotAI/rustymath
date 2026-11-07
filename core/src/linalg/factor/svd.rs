//! The SVD, as LAPACK's `dgebrd`/`zgebrd`, `dorgbr`/`zungbr` and `dbdsqr` compute it. Householder reflections from the left and the right reduce the matrix to a real upper bidiagonal one. The Golub-Kahan form of the QR algorithm then drives its superdiagonal to zero. Each rotation of two columns of the bidiagonal matrix also rotates two rows of Vᵀ, and each rotation of two rows rotates two rows of Uᵀ, so the rotations read contiguous memory, as `eigen`'s do. Each right reflection of a complex matrix maps the conjugate of its row, as in LAPACK's `zgebrd`. The same sweeps then diagonalise its real bidiagonal matrix, rotating complex rows by real rotations. A wide matrix is replaced by its conjugate transpose, which `tall_svd` reduces first where it is tall.
//!
//! Above `SVD_SPLIT` columns, divide and conquer gives the bidiagonal matrix's singular vectors, and the reflections apply to them without forming Q, as LAPACK's `dbdsdc` and `dormbr` do. The recursion, the deflation and the arrow matrix's SVD follow faer's `bidiag_svd.rs`, and share `secular::root` with `eigen`'s divide and conquer. See `THIRD-PARTY.md`.
use super::*;

/// The native thin SVD `a = U Σ Vᴴ` of the finite, nonempty `m`×`n` matrix `a`, which `Tiled::svd` uses off macOS, as `(U, s, V)`. It gives the singular vectors only where `vectors` is true, and `None` where the QR algorithm doesn't converge.
pub(in crate::linalg) fn native<T: Tiled>(a: &[T], [m, n]: [usize; 2], vectors: bool) -> Option<(Vec<T>, Vec<T::Real>, Vec<T>)> where
    T::Real: Tiled<Real = T::Real> {
    if m < n {
        // The conjugate transpose's SVD is V Σ Uᴴ. `tall_svd` reduces it first where it's tall.
        let (u, s, v) = tall_svd(&adjoint(a, m, n), [n, m], vectors)?;
        return Some((v, s, u));
    }
    bidiagonal_svd(a, [m, n], vectors)
}

/// The SVD of the finite `m`×`n` matrix `a`, where `m` is at least `n`, through its bidiagonal form B, as `native` gives it. Singular values descend. With vectors and more than `SVD_SPLIT` columns, divide and conquer gives B's singular vectors, and the reflections then apply to them. Otherwise the QR algorithm rotates the reflections' Q into B's singular vectors. `None` where the QR algorithm doesn't converge.
fn bidiagonal_svd<T: Tiled>(a: &[T], [m, n]: [usize; 2], vectors: bool) -> Option<(Vec<T>, Vec<T::Real>, Vec<T>)> where T::Real: Tiled<Real = T::Real> {
    let mut w = a.to_vec();
    let scale = safe_scale(a);
    if let Some(c) = scale { for x in &mut w { *x = x.mul_real(c) } }
    let (mut d, mut e, left, right) = bidiagonalize(&mut w, [m, n]);
    // The Wilkinson shift and the rotations that chase it form fourth powers of `d` and `e`. Those overflow or underflow where `safe_scale` leaves the matrix alone, unless the largest of `d` and `e` lies between 1 and 2.
    let top = greatest(&d, |x: T::Real| x.abs()).max(greatest(&e, |x: T::Real| x.abs()));
    let unit = if top > T::Real::zero() && top.is_finite() { unit_scale(top) } else { None };
    if let Some(u) = unit { for x in d.iter_mut().chain(&mut e) { *x = *x * u } }
    let unscaled = |d: &[T::Real]| d.iter().map(|&x| [scale, unit].iter().flatten().fold(x, |x, &c| x / c)).collect();
    if vectors && n > SVD_SPLIT {
        // [Bᵀ; 0] = U' [Σ; 0] V'ᵀ, so B's left singular vectors are the columns of V', and its right ones the first n items of the columns of U'.
        e.push(T::Real::zero());
        let (upt, vpt) = conquer(&mut d, &mut e, SVD_SPLIT)?;
        let mut u: Vec<T> = (0..m * n).map(|ij| if ij < n * n { T::from_real(vpt[ij % n * n + ij / n]) } else { T::zero() }).collect();
        left_reflections(&w, &left, [m, n]).apply(&mut u, n);
        let mut v: Vec<T> = (0..n * n).map(|ij| T::from_real(upt[ij % n * (n + 1) + ij / n])).collect();
        blocked(&w, &right, n).apply(&mut v[n..], n);
        return Some((u, unscaled(&d), v));
    }
    let (mut ut, mut vt) = (Vec::new(), Vec::new());
    if vectors {
        ut = transpose(&left_reflections(&w, &left, [m, n]).q(), m, n);
        vt = reflections(&w, &right, n);
    }
    golub_kahan(&mut d, &mut e, &mut ut, &mut vt)?;
    for (k, x) in d.iter_mut().enumerate() { if *x < T::Real::zero() { *x = -*x; if vectors { for y in &mut vt[k * n..(k + 1) * n] { *y = -*y } } } }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[j].partial_cmp(&d[i]).unwrap_or(Ordering::Equal));
    let s = unscaled(&order.iter().map(|&k| d[k]).collect::<Vec<_>>());
    if !vectors { return Some((Vec::new(), s, Vec::new())); }
    let u = (0..m * n).map(|ij| ut[order[ij % n] * m + ij / n]).collect();
    let v = (0..n * n).map(|ij| vt[order[ij % n] * n + ij / n]).collect();
    Some((u, s, v))
}

/// Reduces the `m`×`n` matrix `w`, where `m` is at least `n`, to real upper bidiagonal form, and gives its diagonal, its superdiagonal and the τ of each left and right reflection, each `I - τ v vᴴ` as `reflector` makes it. Each left reflection applies as its conjugate transpose and maps a column. Each right reflection maps the conjugate of a row, then applies from the right. Each left reflection's vector stays below the diagonal in its column, and each right reflection's beyond the superdiagonal in its row, each with its leading 1 left out. Each step makes one pass over the rows below its own. The pass applies both of its reflections to each row, then adds the row's part of the next step's `vᴴ A`. `eigen::summed_rows` splits each pass's rows across threads.
fn bidiagonalize<T: Tiled>(w: &mut [T], [m, n]: [usize; 2]) -> (Vec<T::Real>, Vec<T::Real>, Vec<T>, Vec<T>) {
    let zero = T::Real::zero();
    let (mut d, mut left) = (vec![zero; n], vec![T::zero(); n]);
    let (mut e, mut right) = (vec![zero; n.saturating_sub(1)], vec![T::zero(); n.saturating_sub(1)]);
    if n == 0 { return (d, e, left, right); }
    // Before step k, `column` holds column k from row k down, and `y` the sum over the rows below row k of each one's items after column k, times the conjugate of its item in column k.
    let (mut column, mut next) = (vec![T::zero(); m], vec![T::zero(); m]);
    let (mut y, mut next_y, mut z) = (vec![T::zero(); n], vec![T::zero(); n], vec![T::zero(); n]);
    let mut parts = Vec::new();
    for i in 0..m { column[i] = w[i * n] }
    eigen::summed_rows(&mut w[n..], n, m - 1, &mut y[..n - 1], &mut parts, |i, row, sums| axpy(sums, column[i + 1].conj(), &row[1..]));
    for k in 0..n {
        let width = n - k - 1;
        let x = &mut column[..m - k];
        let x0 = x[0];
        let tau = reflector(x);
        (left[k], d[k]) = (tau, x[0].re());
        let (head, rest) = w.split_at_mut((k + 1) * n);
        let pivot = &mut head[k * n + k + 1..];
        // vᴴ A on the columns after k, where v is 1 at row k, is row k plus `y` times the conjugate of the factor by which `reflector` scaled v. The reflection applies as `I - τ̄ v vᴴ`.
        let z = &mut z[..width];
        if tau != T::zero() {
            z.copy_from_slice(pivot);
            axpy(z, (x0 - x[0]).reciprocal().conj(), &y[..width]);
            axpy(pivot, -tau.conj(), z);
        }
        // The right reflection zeroes row k beyond the superdiagonal. It maps the conjugate of the row.
        if width > 0 { for p in pivot.iter_mut() { *p = p.conj() } }
        let tau_r = if width > 0 { reflector(pivot) } else { T::zero() };
        if width > 0 { (right[k], e[k]) = (tau_r, pivot[0].re()) }
        let sums = &mut next_y[..width.saturating_sub(1)];
        sums.fill(T::zero());
        let (x, z, pivot) = (&*x, &*z, &*pivot);
        eigen::summed_rows(rest, n, m - k - 1, sums, &mut parts, |j, row, sums| {
            row[k] = x[j + 1];
            if width == 0 { return; }
            let r = &mut row[k + 1..];
            if tau != T::zero() { axpy(r, -(tau.conj() * x[j + 1]), z) }
            if tau_r != T::zero() {
                // r ← r (I - τ v vᴴ).
                let s = tau_r * (r[0] + dot(&r[1..], &pivot[1..]));
                r[0] = r[0] - s;
                axpyc(&mut r[1..], -s, &pivot[1..]);
            }
            if j > 0 { axpy(sums, r[0].conj(), &r[1..]) }
        });
        if width > 0 { for (j, row) in rest.chunks_exact(n).enumerate() { next[j] = row[k + 1] } }
        std::mem::swap(&mut column, &mut next);
        std::mem::swap(&mut y, &mut next_y);
    }
    (d, e, left, right)
}

/// The left reflections that `bidiagonalize` leaves in `w`, in QR's blocked form.
fn left_reflections<T: Tiled>(w: &[T], taus: &[T], [m, n]: [usize; 2]) -> Reflections<T> {
    Reflections { m, blocks: blocks_of(&taus[..n], qr_block(n), |k0, k1| transposed(w, n, k0, k1)) }
}

/// Drives the superdiagonal `e` of the real upper bidiagonal matrix with diagonal `d` to zero, leaving its singular values in `d`, some perhaps negative. Each rotation of two of its columns also rotates the same two rows of `vt`, and each rotation of two rows the same rows of `ut`, when they hold vectors. Rows of either may be longer than `n`. `None` where it doesn't converge.
fn golub_kahan<T: Scalar>(d: &mut [T::Real], e: &mut [T::Real], ut: &mut [T], vt: &mut [T]) -> Option<()> {
    let n = d.len();
    if n < 2 { return Some(()); }
    let (m, w, vectors) = (ut.len() / n, vt.len() / n, !vt.is_empty());
    let (zero, eps) = (T::Real::zero(), T::Real::epsilon());
    let tiny = eps * d.iter().chain(e.iter()).fold(zero, |b, &x| b.max(x.abs()));
    let negligible = |e: T::Real, d0: T::Real, d1: T::Real| e.abs() <= eps * (d0.abs() + d1.abs()) || e.abs() <= T::Real::min_positive_value();
    let (mut end, mut sweeps) = (n, 0);
    while end > 1 {
        if negligible(e[end - 2], d[end - 2], d[end - 1]) {
            e[end - 2] = zero;
            end -= 1;
            continue;
        }
        // The unreduced block that ends at `end` starts after the last negligible superdiagonal item before it.
        let mut start = end - 2;
        while start > 0 && !negligible(e[start - 1], d[start - 1], d[start]) { start -= 1 }
        if start > 0 { e[start - 1] = zero }
        sweeps += 1;
        if sweeps > 30 * n { return None; }
        if let Some(i) = (start..end - 1).find(|&i| d[i].abs() <= tiny) {
            // Rotations of row i with each later row zero its superdiagonal item, leaving a singular value of 0.
            let mut f = e[i];
            (d[i], e[i]) = (zero, zero);
            for j in i + 1..end {
                let (c, s, r) = givens(d[j], f);
                d[j] = r;
                if j + 1 < end { (f, e[j]) = (-s * e[j], c * e[j]) }
                if vectors {
                    let (ri, rj) = two_rows(ut, m, i, j);
                    rotate(rj, ri, c, -s);
                }
            }
            continue;
        }
        // The Wilkinson shift: the eigenvalue of BᵀB's trailing 2×2 block nearer its last diagonal item.
        let (a, b) = (end - 2, end - 1);
        let t00 = d[a] * d[a] + if a > start { e[a - 1] * e[a - 1] } else { zero };
        let (t11, t01) = (d[b] * d[b] + e[a] * e[a], d[a] * e[a]);
        let half = (t00 - t11) / (T::Real::one() + T::Real::one());
        let delta = plain_hypot(half, t01);
        let denominator = half + if half < zero { -delta } else { delta };
        let mu = if denominator == zero { t11 } else { t11 - t01 * t01 / denominator };
        let (mut y, mut z) = (d[start] * d[start] - mu, d[start] * e[start]);
        for k in start..end - 1 {
            // A rotation of columns k and k + 1 zeroes z, the bulge at (k - 1, k + 1), or starts the chase.
            let (c, s, r) = givens(y, z);
            if k > start { e[k - 1] = r }
            (d[k], e[k]) = (c * d[k] + s * e[k], c * e[k] - s * d[k]);
            let bulge = s * d[k + 1];
            d[k + 1] = c * d[k + 1];
            if vectors {
                let (p, q) = two_rows(vt, w, k, k + 1);
                rotate(p, q, c, -s);
            }
            // A rotation of rows k and k + 1 zeroes the bulge at (k + 1, k), and makes one at (k, k + 2).
            let (c, s, r) = givens(d[k], bulge);
            d[k] = r;
            (e[k], d[k + 1]) = (c * e[k] + s * d[k + 1], c * d[k + 1] - s * e[k]);
            if k + 2 < end { (y, z, e[k + 1]) = (e[k], s * e[k + 1], c * e[k + 1]) }
            if vectors {
                let (p, q) = two_rows(ut, m, k, k + 1);
                rotate(p, q, c, -s);
            }
        }
    }
    Some(())
}

/// The plane rotation `(c, s)` that takes `(p, q)` to `(r, 0)`, with `c p - s q = r` and `s p + c q = 0`. `rotate` applies it to two rows.
fn plane<R: Native>(p: R, q: R) -> (R, R) {
    let (zero, one) = (R::zero(), R::one());
    if q == zero { return (if p < zero { -one } else { one }, zero); }
    if p == zero { return (zero, if q < zero { one } else { -one }); }
    let (big, small, flip) = if p.abs() > q.abs() { (p, q, false) } else { (q, p, true) };
    let t = small / big;
    let u = t.mla::<FUSED>(t, one).sqrt();
    let r = (if big < zero { -u } else { u }).recip();
    if flip { (t * r, -r) } else { (r, -t * r) }
}

/// The SVD of the (n+1)×n lower bidiagonal matrix with diagonal `d` and subdiagonal `sub`, each of n items, by divide and conquer. Its singular values replace `d`, in descending order, and `sub` is overwritten. It gives Uᵀ, (n+1)×(n+1), and Vᵀ, n×n, each holding one singular vector in each row. A matrix of at most `leaf` columns, or 2, is a leaf, which `qr_leaf` solves. `None` where the QR algorithm doesn't converge in a leaf.
///
/// Column k splits the matrix. Its diagonal item α and subdiagonal item β leave a (k+1)×k matrix above and to the left of them, and a (rem+1)×rem one below and to the right. In the halves' singular vectors, the matrix is an arrow matrix M: a first column z, and a diagonal of the halves' singular values. M's SVD comes from one secular equation for each singular value, as in `eigen::merge`.
fn conquer<R: Native + Tiled<Real = R>>(d: &mut [R], sub: &mut [R], leaf: usize) -> Option<(Vec<R>, Vec<R>)> {
    let n = d.len();
    if n <= leaf.max(2) { return qr_leaf(d, sub); }
    let zero = R::zero();
    let largest = d.iter().chain(sub.iter()).fold(zero, |m, x| m.max(x.abs()));
    if largest == zero { return Some((identity(n + 1), identity(n))); }
    for x in d.iter_mut().chain(sub.iter_mut()) { *x = *x / largest }
    let (k, rem) = (n / 2, n - n / 2 - 1);
    let (alpha, beta) = (d[k], sub[k]);
    let (u1, v1) = conquer(&mut d[..k], &mut sub[..k], leaf)?;
    let (u2, v2) = conquer(&mut d[k + 1..], &mut sub[k + 1..], leaf)?;
    // q1 and q2, the last rows of u1 and u2, are the halves' left singular vectors with no singular value.
    let (w1, w2) = (k + 1, rem + 1);
    let (q1, q2) = (&u1[k * w1..], &u2[rem * w2..]);
    let (al, bp) = (alpha * q1[k], beta * q2[0]);
    let r0 = al.mla::<FUSED>(al, bp * bp).sqrt();
    let (c0, s0) = if r0 == zero { (R::one(), zero) } else { (al / r0, bp / r0) };
    // M's diagonal: r0, then the halves' singular values. Its first column z: r0, then α times the last item of each of the upper half's left singular vectors, then β times the first item of each of the lower half's.
    for i in (0..k).rev() { d[i + 1] = d[i] }
    d[0] = r0;
    let z = sub;
    z[0] = r0;
    for j in 1..=k { z[j] = alpha * u1[(j - 1) * w1 + k] }
    for j in k + 1..n { z[j] = beta * u2[(j - k - 1) * w2] }
    let (outer, turns0, turns) = deflate(d, z, k);
    let (mut um, mut vm) = arrow(d, z, &outer);
    // Undo the deflating rotations: each turned two rows of M, so it turns the same items of M's singular vectors back.
    let turn = |rows: &mut [R], width: usize, (a, b): (usize, usize), (c, s): (R, R)| {
        for row in rows.chunks_exact_mut(width).take(n) { (row[b], row[a]) = (c * row[b] - s * row[a], c * row[a] + s * row[b]) }
    };
    for &(i, c, s) in turns.iter().rev() {
        let rows = (outer[i - 1], outer[i]);
        turn(&mut um, n + 1, rows, (c, s));
        turn(&mut vm, n, rows, (c, s));
    }
    for &(i, c, s) in turns0.iter().rev() { turn(&mut um, n + 1, (0, i), (c, s)) }
    // The whole matrix's left singular vectors are M's, times the halves' put together. U's column 0 is c0 q1 over s0 q2, its column n is -s0 q1 over c0 q2, and its other columns are the halves' vectors.
    let mut ut = vec![zero; (n + 1) * (n + 1)];
    let b1: Vec<R> = q1.iter().map(|&q| c0 * q).chain(u1[..k * w1].iter().copied()).collect();
    let mut b2 = u2.clone();
    for x in &mut b2[rem * w2..] { *x = c0 * *x }
    let um_view = Matrix::rows(&um, n + 1);
    R::gemm(&mut ut, n + 1, um_view, Matrix::rows(&b1, w1), [n + 1, w1, w1], R::one(), zero);
    R::gemm(&mut ut[w1..], n + 1, um_view.offset(0, w1), Matrix::rows(&b2, w2), [n + 1, w2, w2], R::one(), zero);
    for (row, m) in ut.chunks_exact_mut(n + 1).zip(um.chunks_exact(n + 1)) {
        let (top, bottom) = row.split_at_mut(w1);
        axpy(top, -s0 * m[n], q1);
        axpy(bottom, s0 * m[0], q2);
    }
    // V's column 0 is the unit vector at row k, and its other columns are the halves' vectors.
    let mut vt = vec![zero; n * n];
    R::gemm(&mut vt, n, Matrix::rows(&vm, n).offset(0, 1), Matrix::rows(&v1, k), [n, k, k], R::one(), zero);
    R::gemm(&mut vt[w1..], n, Matrix::rows(&vm, n).offset(0, w1), Matrix::rows(&v2, rem), [n, rem, rem], R::one(), zero);
    for (row, m) in vt.chunks_exact_mut(n).zip(vm.chunks_exact(n)) { row[k] = m[0] }
    for x in d.iter_mut() { *x = *x * largest }
    Some((ut, vt))
}

/// `conquer` for a leaf of the recursion. Rotations of the last row with each row above it, from the bottom, move its subdiagonal item up and out, leaving a square lower bidiagonal matrix. That matrix is the transpose of an upper one, whose row rotations act on V, and whose column rotations act on the rows of Uᵀ.
fn qr_leaf<R: Native + Tiled<Real = R>>(d: &mut [R], sub: &mut [R]) -> Option<(Vec<R>, Vec<R>)> {
    let (n, w) = (d.len(), d.len() + 1);
    let (mut ut, mut vt) = (identity(w), identity(n));
    let mut bulge = sub[n - 1];
    sub[n - 1] = R::zero();
    for i in (0..n).rev() {
        let (c, s) = plane(d[i], bulge);
        d[i] = c * d[i] - s * bulge;
        if i > 0 { (bulge, sub[i - 1]) = (s * sub[i - 1], c * sub[i - 1]) }
        let (row, last) = two_rows(&mut ut, w, i, n);
        rotate(row, last, c, s);
    }
    golub_kahan::<R>(d, &mut sub[..n - 1], &mut vt, &mut ut[..n * w])?;
    for (k, x) in d.iter_mut().enumerate() { if *x < R::zero() { *x = -*x; for y in &mut vt[k * n..(k + 1) * n] { *y = -*y } } }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[j].partial_cmp(&d[i]).unwrap_or(Ordering::Equal));
    let (d0, u0, v0) = (d.to_vec(), ut.clone(), vt.clone());
    for (k, &o) in order.iter().enumerate() {
        d[k] = d0[o];
        ut[k * w..(k + 1) * w].copy_from_slice(&u0[o * w..(o + 1) * w]);
        vt[k * n..(k + 1) * n].copy_from_slice(&v0[o * n..(o + 1) * n]);
    }
    Some((ut, vt))
}

/// Deflates the arrow matrix with diagonal `d` and first column `z`, whose items 1 to `k` and `k + 1` onward each descend. A negligible item of z leaves its singular value and vectors unchanged. A rotation moves into `z[0]` the part of z at a negligible singular value, and into `z[i - 1]` the part of z at a singular value equal to its neighbour's. It sorts the items, and gives the permutation of M's rows that it applied, then the rotations of each kind as `(i, c, s)`.
fn deflate<R: Native>(d: &mut [R], z: &mut [R], k: usize) -> (Vec<usize>, Vec<(usize, R, R)>, Vec<(usize, R, R)>) {
    let (n, zero) = (d.len(), R::zero());
    let largest = |x: &[R]| x.iter().fold(zero, |m, v| m.max(v.abs()));
    let (max_d, max_z) = (largest(&d[1..]), largest(z));
    let (eps, small) = (R::epsilon(), R::min_positive_value());
    let strict = (eps * max_d).max(small);
    let coarse = R::narrow(8.0) * eps * max_d.max(max_z);
    if d[0] < coarse { (d[0], z[0]) = (coarse, coarse) }
    for x in &mut z[1..] { if x.abs() < strict { *x = zero } }
    let mut turns0 = Vec::new();
    for i in 1..n {
        let (p, q) = (z[0], z[i]);
        if d[i] < coarse && (p != zero || q != zero) {
            let (c, s) = plane(p, q);
            let r = c * p - s * q;
            (z[0], d[0], z[i]) = (r, r, zero);
            turns0.push((i, c, s));
        }
    }
    let total = z[1..].iter().all(|x| x.abs() < small);
    // `perm` lists the zero singular values, then merges the halves' into descending order.
    let mut perm = vec![0; n];
    let mut p = 1;
    for i in 1..n {
        if d[i].abs() < small {
            perm[p] = i;
            p += 1;
        }
    }
    let (mut i, mut j) = (1, k + 1);
    for slot in &mut perm[p..] {
        if i > k || (j < n && d[i] < d[j]) {
            *slot = j;
            j += 1;
        } else {
            *slot = i;
            i += 1;
        }
    }
    if total {
        for i in 1..n { if d[perm[i]].abs() < small || d[perm[i]] > d[0] { perm[i - 1] = perm[i] } else { perm[i - 1] = 0; break; } }
    }
    // Swaps put d into ascending order, from the end of `perm`, recording each swap.
    let (mut real_ind, mut real_col, mut swaps): (Vec<usize>, Vec<usize>, Vec<usize>) = ((0..n).collect(), (0..n).collect(), (0..n).collect());
    let start = if total { 0 } else { 1 };
    for i in start..n {
        let pi = perm[n - i - 1 + start];
        let j = real_col[pi];
        d.swap(i, j);
        if i != 0 && j != 0 { z.swap(i, j) }
        swaps[i] = j;
        let real_i = real_ind[i];
        (real_col[real_i], real_col[pi], real_ind[j], real_ind[i]) = (j, i, real_i, pi);
    }
    z[0] = d[0];
    let mut outer: Vec<usize> = (0..n).collect();
    for (i, &j) in swaps.iter().enumerate() { outer.swap(i, j) }
    let mut i = n - 1;
    while i > 0 && (d[i].abs() < small || z[i].abs() < small) { i -= 1 }
    let mut turns = Vec::new();
    while i > 1 {
        if d[i] - d[i - 1] < strict {
            let (p, q) = (z[i - 1], z[i]);
            if p != zero || q != zero {
                let (c, s) = plane(p, q);
                (z[i - 1], z[i]) = (c * p - s * q, zero);
                turns.push((i, c, s));
            }
            d[i - 1] = d[i];
        }
        i -= 1;
    }
    (outer, turns0, turns)
}

/// The SVD of the deflated n×n arrow matrix M with diagonal `d` and first column `z`, where `d[0]` equals `z[0]`. Its singular values replace `d`, in descending order. It gives the rows of Uᵀ, (n+1)×(n+1), whose last row is the unit vector at item n, and of Vᵀ, n×n, with their items in the order of M's rows before `deflate` permuted them by `outer`.
fn arrow<R: Native + Tiled<Real = R>>(d: &mut [R], z: &[R], outer: &[usize]) -> (Vec<R>, Vec<R>) {
    let (n, zero, one) = (d.len(), R::zero(), R::one());
    d[0] = zero;
    let mut actual = n;
    while actual > 1 && d[actual - 1] == zero { actual -= 1 }
    let perm: Vec<usize> = (0..actual).filter(|&i| z[i] != zero).collect();
    let (zp, dp): (Vec<R>, Vec<R>) = perm.iter().map(|&i| (z[i], d[i])).unzip();
    // Each singular value σ is a root of the secular equation 1 + Σ zᵢ²/((dᵢ - σ)(dᵢ + σ)) = 0, between consecutive dᵢ, or above the last.
    let f = |shift: R, mu: R| zp.iter().zip(&dp).fold(one, |s, (&c, &di)| s + (c / ((di - shift) - mu)) * (c / ((di + shift) + mu)));
    let four = |shift: [R; 4], mu: [R; 4]| {
        let mut s = [one; 4];
        for (&c, &di) in zp.iter().zip(&dp) { for t in 0..4 { s[t] = s[t] + c * c / (((di - shift[t]) - mu[t]) * ((di + shift[t]) + mu[t])) } }
        s
    };
    let mut last = n;
    while last > 1 && z[last - 1] == zero { last -= 1 }
    let norm = z.iter().fold(zero, |s, &x| x.mla::<FUSED>(x, s)).sqrt();
    let mut roots = vec![(zero, zero, zero); n];
    crate::parallel::split(&mut roots, 1, &|first, out| {
        for (j, root) in out.iter_mut().enumerate() {
            let k = first + j;
            *root = if z[k] == zero || last == 1 { let s = if k == 0 { z[0] } else { d[k] }; (s, s, zero) } else {
                let end = k == last - 1;
                let right = if end { d[last - 1] + norm } else { d[(k + 1..n).find(|&l| z[l] != zero).expect("a later item of z is nonzero")] };
                let (shift, mu) = secular::root(&f, &four, d[k], right, end);
                (shift + mu, shift, mu)
            };
        }
        true
    });
    let mut s: Vec<R> = roots.iter().map(|r| r.0).collect();
    let shifts: Vec<R> = roots.iter().map(|r| r.1).collect();
    let mus: Vec<R> = roots.iter().map(|r| r.2).collect();
    // Gu and Eisenstat's z: the vector for which the computed singular values are exact. Its singular vectors come out orthogonal.
    let mut zhat = vec![zero; n];
    if let Some(&tail) = perm.last() {
        crate::parallel::split(&mut zhat, 1, &|first, out| {
            for (j, zk) in out.iter_mut().enumerate() {
                let k = first + j;
                if z[k] == zero { continue; }
                let dk = d[k];
                let mut product = (s[tail] + dk) * (mus[tail] + (shifts[tail] - dk));
                for (l, &i) in perm.iter().enumerate().filter(|&(_, &i)| i != k) {
                    if i >= k && l == 0 { product = zero; break; }
                    let j = if i < k || l == 0 { i } else { perm[l - 1] };
                    product = product * (((s[j] + dk) / (d[i] + dk)) * ((mus[j] + (shifts[j] - dk)) / (d[i] - dk)));
                }
                *zk = if z[k] > zero { product.sqrt() } else { -product.sqrt() };
            }
            true
        });
    }
    let mut order: Vec<usize> = (0..actual).collect();
    for i in 1..actual {
        if s[i - 1] > s[i] {
            s.swap(i - 1, i);
            order.swap(i - 1, i);
        }
    }
    let rank: Vec<usize> = crate::search::inverse_permutation(order.iter().copied());
    let mut from = vec![0; n];
    for k in 0..n { from[if k < actual { actual - rank[k] - 1 } else { k }] = k }
    let (mut ut, mut vt) = (vec![zero; (n + 1) * (n + 1)], vec![zero; n * n]);
    let unit = |x: &mut [R]| {
        let norm = x.iter().fold(zero, |s, &v| v.mla::<FUSED>(v, s)).sqrt();
        for v in x.iter_mut() { *v = *v / norm }
    };
    {
        let mut rows: Vec<(&mut [R], &mut [R])> = ut.chunks_exact_mut(n + 1).zip(vt.chunks_exact_mut(n)).collect();
        crate::parallel::split(&mut rows, 1, &|start, pairs| {
            for (row, (u, v)) in (start..).zip(pairs.iter_mut()) {
                let (u, k) = (&mut u[..n], from[row]);
                if zhat[k] == zero { (u[outer[k]], v[outer[k]]) = (one, one); continue; }
                let (shift, mu) = (shifts[k], mus[k]);
                for &i in &perm { u[outer[i]] = zhat[i] / ((d[i] - shift) - mu) / (d[i] + (shift + mu)) }
                unit(u);
                v[outer[0]] = -one;
                for &i in &perm[1..] { v[outer[i]] = d[i] * zhat[i] / ((d[i] - shift) - mu) / (d[i] + (shift + mu)) }
                unit(v);
            }
            true
        });
    }
    ut[n * (n + 1) + n] = one;
    for i in 0..n { d[i] = if i < actual { s[actual - i - 1] } else { s[i] } }
    (ut, vt)
}
