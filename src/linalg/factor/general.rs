//! The general eigen problem for real and complex matrices, as LAPACK's `dgehd2`, `dlahqr` and `dtrevc` solve it for real ones, and `zgehd2`, `zlahqr` and `ztrevc` for complex ones. Householder reflections reduce the matrix to upper Hessenberg form. QR iterations then reduce that to Schur form. For a real matrix, Francis double-shift iterations leave a quasi-triangular matrix whose 2×2 diagonal blocks each hold a complex conjugate pair of eigenvalues. For a complex matrix, single-shift iterations leave a triangular one. Back substitution gives the Schur form's eigenvectors. The accumulated transformations turn them into the matrix's own. Those transformations build up as the rows of Zᵀ, where each reflection reads contiguous memory.
use super::*;
use num_complex::Complex;
use std::ops::Range;

/// The native eigen decomposition of the finite, nonempty real or complex `n`×`n` matrix `a`, with eigenvectors only where `vectors` is true. `Tiled::eig` uses it for complex matrices on every target, and for real ones off macOS. A matrix whose largest item lies outside the safe range first scales by a power of two. Its eigenvalues then scale back exactly.
pub(in crate::linalg) fn native<T: Schur>(a: &[T], n: usize, vectors: bool) -> Option<Eig<T::Real>> {
    // The working copy's rows start `ld` items apart. The right reflections read down columns. With rows whose length is a multiple of a large power of two, a column's items would all fall in the same few cache sets and keep evicting each other.
    let ld = n + 8;
    let mut h = vec![T::zero(); n * ld];
    for (row, src) in h.chunks_exact_mut(ld).zip(a.chunks_exact(n)) { row[..n].copy_from_slice(src) }
    let scale = safe_scale(a);
    if let Some(s) = scale { for x in &mut h { *x = x.mul_real(s) } }
    let (w, taus) = hessenberg(&mut h, n, ld);
    let mut zt = if vectors { reflections(&w, &taus, n) } else { Vec::new() };
    let schur = T::schur(&mut h, ld, &mut zt, n, vectors)?;
    let values = schur.iter().map(|&z| scale.map_or(z, |s| z / s)).collect();
    if !vectors { return Some(Eig { values, vectors: Vec::new() }); }
    let x = T::schur_vectors(&h, ld, n, &schur);
    let v = T::multiply(Matrix::rows(&zt, n).transpose(), Matrix::rows(&x, n), [n; 3]);
    Some(Eig { values, vectors: T::unit_vectors(&v, n, &schur) })
}

/// The steps of the general eigen problem that differ between real and complex matrices. `native` runs the steps they share.
pub(in crate::linalg) trait Schur: Tiled {
    /// Reduces the upper Hessenberg matrix `h`, whose rows start `ld` items apart, to Schur form, and gives the eigenvalues in the order the Schur form holds them. Where `wantt` is true, each iteration transforms the whole of `h`. Each transformation then also acts on the rows of `zt`, which hold Qᵀ on entry and Zᵀ on exit. Where `wantt` is false, `zt` is empty. Each iteration then transforms only the rows and columns it still acts on. `None` where the iterations don't converge.
    fn schur(h: &mut [Self], ld: usize, zt: &mut [Self], n: usize, wantt: bool) -> Option<Vec<Complex<Self::Real>>>;
    /// The eigenvectors of the Schur form `t`, whose rows start `ld` items apart, as the columns of an `n`×`n` matrix, for the eigenvalues `values` that `schur` gave.
    fn schur_vectors(t: &[Self], ld: usize, n: usize, values: &[Complex<Self::Real>]) -> Vec<Self>;
    /// The eigenvectors in the columns of the `n`×`n` matrix `v`, each scaled to unit length.
    fn unit_vectors(v: &[Self], n: usize, values: &[Complex<Self::Real>]) -> Vec<Complex<Self::Real>>;
}

/// A real matrix's Schur form is quasi-triangular. Each 2×2 diagonal block holds a complex conjugate pair of eigenvalues.
impl<T: Native + Tiled<Real = T>> Schur for T {
    fn schur(h: &mut [T], ld: usize, zt: &mut [T], n: usize, wantt: bool) -> Option<Vec<Complex<T>>> {
        let (wr, wi) = francis(h, ld, zt, n, wantt)?;
        Some(wr.into_iter().zip(wi).map(|(re, im)| Complex::new(re, im)).collect())
    }
    fn schur_vectors(t: &[T], ld: usize, n: usize, values: &[Complex<T>]) -> Vec<T> { quasi_vectors(t, ld, n, values) }
    fn unit_vectors(v: &[T], n: usize, values: &[Complex<T>]) -> Vec<Complex<T>> { real_units(v, n, values) }
}

/// A complex matrix's Schur form is upper triangular, with the eigenvalues on its diagonal.
impl<R: Native + Tiled<Real = R>> Schur for Complex<R> where Complex<R>: Tiled<Real = R> {
    fn schur(h: &mut [Self], ld: usize, zt: &mut [Self], n: usize, wantt: bool) -> Option<Vec<Self>> { single_shift(h, ld, zt, n, wantt) }
    fn schur_vectors(t: &[Self], ld: usize, n: usize, _: &[Self]) -> Vec<Self> { triangular_vectors(t, ld, n) }
    fn unit_vectors(v: &[Self], n: usize, _: &[Self]) -> Vec<Self> {
        let mut vectors = v.to_vec();
        for j in 0..n {
            let scale = unit(&(0..n).map(|i| v[i * n + j]).collect::<Vec<_>>());
            for i in 0..n { vectors[i * n + j] = vectors[i * n + j].mul_real(scale) }
        }
        vectors
    }
}

/// Reduces `h` to upper Hessenberg form by Householder reflections, as LAPACK's `dgehd2` and `zgehd2` do, and gives the reflections in the layout that `reflections` reads, with each one's `τ`. Reflection `k`, `H = I - τ v vᴴ`, maps column `k`'s items below the subdiagonal to zero. Its vector is 1 at row `k + 1`, and row `k` of the first result keeps the rest of it, from column `k + 2`. Each reflection applies as `Hᴴ` from the left to the rows below row `k`, then as `H` from the right to every row.
fn hessenberg<T: Scalar>(h: &mut [T], n: usize, ld: usize) -> (Vec<T>, Vec<T>) {
    let (mut w, mut taus) = (vec![T::zero(); n * n], vec![T::zero(); n.saturating_sub(1)]);
    let (mut v, mut y) = (vec![T::zero(); n], vec![T::zero(); n]);
    for k in 0..n.saturating_sub(2) {
        let m = n - k - 1;
        let v = &mut v[..m];
        for (i, x) in v.iter_mut().enumerate() { *x = h[(k + 1 + i) * ld + k] }
        let tau = reflector(v);
        h[(k + 1) * ld + k] = v[0];
        for i in 1..m { h[(k + 1 + i) * ld + k] = T::zero() }
        if tau == T::zero() { continue; }
        taus[k] = tau;
        v[0] = T::one();
        w[k * n + k + 2..(k + 1) * n].copy_from_slice(&v[1..]);
        // From the left, on the rows below row k, from column k + 1: B ← B - τ̄ v (vᴴ B).
        let span = |i: usize| (k + 1 + i) * ld + k + 1..(k + 1 + i) * ld + n;
        let y = &mut y[..m];
        y.copy_from_slice(&h[span(0)]);
        for i in 1..m { axpy(y, v[i].conj(), &h[span(i)]) }
        for i in 0..m { axpy(&mut h[span(i)], -(tau.conj() * v[i]), y) }
        // From the right, on every row from column k + 1.
        for row in h.chunks_exact_mut(ld) {
            let row = &mut row[k + 1..n];
            let s = tau * dot(row, v);
            axpyc(row, -s, v);
        }
    }
    (w, taus)
}

/// Reduces the upper Hessenberg matrix `h`, whose rows start `ld` items apart, to real Schur form by Francis double-shift QR iterations, as LAPACK's `dlahqr` does. Gives the real and imaginary parts of the eigenvalues. Where `wantt` is true, each iteration transforms the whole of `h`. Each reflection and rotation then also acts on the rows of `zt`, which hold Qᵀ on entry and Zᵀ on exit. Where `wantt` is false, `zt` is empty. Each iteration then transforms only the rows and columns it still acts on. Each converged 2×2 block takes the standard form that `standard` gives. `None` where a block takes more than `30 max(10, n)` iterations.
fn francis<T: Native>(h: &mut [T], ld: usize, zt: &mut [T], n: usize, wantt: bool) -> Option<(Vec<T>, Vec<T>)> {
    let (zero, one, two) = (T::zero(), T::one(), T::one() + T::one());
    let (mut wr, mut wi) = (vec![zero; n], vec![zero; n]);
    if n == 0 { return Some((wr, wi)); }
    let (ulp, safmin) = (T::epsilon(), T::min_positive_value());
    let smlnum = safmin * (T::narrow(n as f64) / ulp);
    let (dat1, dat2) = (T::narrow(0.75), T::narrow(-0.4375));
    let itmax = 30 * n.max(10);
    let (mut i1, mut i2) = (0, n - 1);
    // Iterations since the last deflation. Every tenth one takes an exceptional shift.
    let mut kdefl = 0;
    let mut v = [zero; 3];
    let mut i = n - 1;
    loop {
        // Rows and columns l to i form the active block. Eigenvalues after i have converged.
        let mut l = 0;
        let mut its = 0;
        loop {
            // The last negligible subdiagonal item at or above row i splits off the block that holds row i.
            let mut k = i;
            while k > l {
                let sub = h[k * ld + k - 1].abs();
                if sub <= smlnum { break; }
                let mut tst = h[(k - 1) * ld + k - 1].abs() + h[k * ld + k].abs();
                if tst == zero { if k >= 2 { tst = tst + h[(k - 1) * ld + k - 2].abs() } if k + 1 < n { tst = tst + h[(k + 1) * ld + k].abs() } }
                // Ahues and Tisseur's conservative test, as LAPACK applies it.
                if sub <= ulp * tst {
                    let sup = h[(k - 1) * ld + k].abs();
                    let (ab, ba) = (sub.max(sup), sub.min(sup));
                    let (d, e) = (h[k * ld + k].abs(), (h[(k - 1) * ld + k - 1] - h[k * ld + k]).abs());
                    let (aa, bb) = (d.max(e), d.min(e));
                    let s = aa + ab;
                    if ba * (ab / s) <= smlnum.max(ulp * (bb * (aa / s))) { break; }
                }
                k -= 1;
            }
            l = k;
            if l > 0 { h[l * ld + l - 1] = zero }
            if l + 1 >= i { break; }
            its += 1;
            if its > itmax { return None; }
            kdefl += 1;
            if !wantt { (i1, i2) = (l, i) }
            let (h11, h12, h21, h22) = if kdefl % 20 == 0 {
                let s = h[i * ld + i - 1].abs() + h[(i - 1) * ld + i - 2].abs();
                (dat1 * s + h[i * ld + i], dat2 * s, s, dat1 * s + h[i * ld + i])
            } else if kdefl % 10 == 0 {
                let s = h[(l + 1) * ld + l].abs() + h[(l + 2) * ld + l + 1].abs();
                (dat1 * s + h[l * ld + l], dat2 * s, s, dat1 * s + h[l * ld + l])
            } else { (h[(i - 1) * ld + i - 1], h[(i - 1) * ld + i], h[i * ld + i - 1], h[i * ld + i]) };
            // The shifts are the eigenvalues of the 2×2 block above, or the real one nearer h22 twice.
            let s = h11.abs() + h12.abs() + h21.abs() + h22.abs();
            let (rt1r, rt1i, rt2r, rt2i) = if s == zero { (zero, zero, zero, zero) } else {
                let (h11, h12, h21, h22) = (h11 / s, h12 / s, h21 / s, h22 / s);
                let tr = (h11 + h22) / two;
                let det = (h11 - tr) * (h22 - tr) - h12 * h21;
                let rtdisc = det.abs().sqrt();
                if det >= zero { (tr * s, rtdisc * s, tr * s, -rtdisc * s) } else {
                    let (r1, r2) = (tr + rtdisc, tr - rtdisc);
                    let r = if (r1 - h22).abs() <= (r2 - h22).abs() { r1 * s } else { r2 * s };
                    (r, zero, r, zero)
                }
            };
            // The sweep starts at the lowest row m where two consecutive small subdiagonal items let it begin without disturbing the rows above. v is the first column of the shifted product, scaled.
            let mut m = i - 2;
            loop {
                let (hmm, hm1m, hmm1) = (h[m * ld + m], h[(m + 1) * ld + m], h[m * ld + m + 1]);
                let s = (hmm - rt2r).abs() + rt2i.abs() + hm1m.abs();
                let h21s = hm1m / s;
                v[0] = h21s * hmm1 + (hmm - rt1r) * ((hmm - rt2r) / s) - rt1i * (rt2i / s);
                v[1] = h21s * (hmm + h[(m + 1) * ld + m + 1] - rt1r - rt2r);
                v[2] = h21s * h[(m + 2) * ld + m + 1];
                let s = v[0].abs() + v[1].abs() + v[2].abs();
                for x in &mut v { *x = *x / s }
                if m == l { break; }
                let h00 = h[m * ld + m - 1].abs() * (v[1].abs() + v[2].abs());
                let h01 = ulp * v[0].abs() * (h[(m - 1) * ld + m - 1].abs() + hmm.abs() + h[(m + 1) * ld + m + 1].abs());
                if h00 <= h01 { break; }
                m -= 1;
            }
            // The double-shift sweep: a reflection of order 3 makes a bulge, and each later one chases it down a row.
            for k in m..i {
                let nr = 3.min(i - k + 1);
                if k > m { for r in 0..nr { v[r] = h[(k + r) * ld + k - 1] } }
                let t1 = reflector(&mut v[..nr]);
                if k > m {
                    h[k * ld + k - 1] = v[0];
                    h[(k + 1) * ld + k - 1] = zero;
                    if k + 1 < i { h[(k + 2) * ld + k - 1] = zero }
                } else if m > l {
                    // This avoids LAPACK's bug where v[1] and v[2] underflow.
                    h[k * ld + k - 1] = h[k * ld + k - 1] * (one - t1);
                }
                let three = nr == 3;
                let bulge = Bulge { k, three, v: [v[1], if three { v[2] } else { zero }], t: [t1, t1 * v[1], if three { t1 * v[2] } else { zero }] };
                bulge.left(h, ld, k..=i2);
                if wantt { bulge.left(zt, n, 0..=n - 1) }
                for row in h[i1 * ld..(i.min(k + 3) + 1) * ld].chunks_exact_mut(ld) { bulge.right(row) }
            }
        }
        if l == i { wr[i] = h[i * ld + i]; } else {
            // A 2×2 block in rows p and q has converged.
            let (p, q) = (i - 1, i);
            let (block, values, (cs, sn)) = standard(h[p * ld + p], h[p * ld + q], h[q * ld + p], h[q * ld + q]);
            [h[p * ld + p], h[p * ld + q], h[q * ld + p], h[q * ld + q]] = block;
            [wr[p], wi[p], wr[q], wi[q]] = values;
            if wantt {
                // The block's rotation acts on rows p and q to the right of it, on columns p and q above it, and on rows p and q of Zᵀ.
                if i2 > q {
                    let (x, y) = two_rows(h, ld, p, q);
                    rotate(&mut x[q + 1..=i2], &mut y[q + 1..=i2], cs, -sn);
                }
                for row in h[i1 * ld..p * ld].chunks_exact_mut(ld) { (row[p], row[q]) = (cs * row[p] + sn * row[q], cs * row[q] - sn * row[p]) }
                let (x, y) = two_rows(zt, n, p, q);
                rotate(x, y, cs, -sn);
            }
        }
        kdefl = 0;
        if l == 0 { return Some((wr, wi)); }
        i = l - 1;
    }
}

/// One reflection of a double-shift sweep, `I - τ u uᵀ`, acting on rows or columns `k` to `k + 2` with `u = [1, v[0], v[1]]`, or on `k` and `k + 1` with `u = [1, v[0]]` where `three` is false. `t` is `τ u`.
#[derive(Clone, Copy)]
struct Bulge<T> { k: usize, three: bool, v: [T; 2], t: [T; 3] }

impl<T: Native> Bulge<T> {
    /// Applies the reflection from the left to `columns` of the matrix `a`, whose rows hold `n` items. Each row's items are contiguous, and the loop over them vectorizes.
    #[inline(always)]
    fn left(&self, a: &mut [T], n: usize, columns: std::ops::RangeInclusive<usize>) {
        let (k, [v2, v3], [t1, t2, t3]) = (self.k, self.v, self.t);
        let (r0, rest) = a[k * n..(k + if self.three { 3 } else { 2 }) * n].split_at_mut(n);
        if self.three {
            let (r1, r2) = rest.split_at_mut(n);
            for ((a, b), c) in r0[columns.clone()].iter_mut().zip(&mut r1[columns.clone()]).zip(&mut r2[columns]) {
                let sum = *a + v2 * *b + v3 * *c;
                (*a, *b, *c) = (*a - sum * t1, *b - sum * t2, *c - sum * t3);
            }
        } else {
            for (a, b) in r0[columns.clone()].iter_mut().zip(&mut rest[columns]) {
                let sum = *a + v2 * *b;
                (*a, *b) = (*a - sum * t1, *b - sum * t2);
            }
        }
    }

    /// Applies the reflection from the right to one row.
    #[inline(always)]
    fn right(&self, row: &mut [T]) {
        let (k, [v2, v3], [t1, t2, t3]) = (self.k, self.v, self.t);
        if self.three {
            let [a, b, c] = &mut row[k..k + 3] else { unreachable!("three items") };
            let sum = *a + v2 * *b + v3 * *c;
            (*a, *b, *c) = (*a - sum * t1, *b - sum * t2, *c - sum * t3);
        } else {
            let [a, b] = &mut row[k..k + 2] else { unreachable!("two items") };
            let sum = *a + v2 * *b;
            (*a, *b) = (*a - sum * t1, *b - sum * t2);
        }
    }
}

/// The standard form of the 2×2 block `[a b; c d]`, as LAPACK's `dlanv2` gives it, then its eigenvalues as `[re1, im1, re2, im2]`, then the rotation `(cs, sn)` that takes it there: `[a b; c d] = [cs -sn; sn cs] F [cs sn; -sn cs]`, where F is the standard form. A complex pair's form has equal diagonal items and off-diagonal items of opposite signs, and `im1 > 0`. A real pair's form is upper triangular.
fn standard<T: Native>(mut a: T, mut b: T, mut c: T, mut d: T) -> ([T; 4], [T; 4], (T, T)) {
    let (zero, one) = (T::zero(), T::one());
    let half = one / (one + one);
    let sign = |x: T| if x.is_sign_negative() { -one } else { one };
    let eps = T::epsilon();
    let safmn2 = T::narrow(2f64.powi(((T::min_positive_value() / eps).wide().log2() / 2.0) as i32));
    let safmx2 = one / safmn2;
    let (mut cs, mut sn) = (one, zero);
    if c == zero { }
    else if b == zero {
        // Swap the rows and columns.
        (cs, sn) = (zero, one);
        (a, d, b, c) = (d, a, -c, zero);
    } else if a - d == zero && sign(b) != sign(c) { } else {
        let mut temp = a - d;
        let p = half * temp;
        let bcmax = b.abs().max(c.abs());
        let bcmis = b.abs().min(c.abs()) * sign(b) * sign(c);
        let scale = p.abs().max(bcmax);
        let z = (p / scale) * p + (bcmax / scale) * bcmis;
        if z >= T::narrow(4.0) * eps {
            // Real eigenvalues: compute a and d.
            let z = p + sign(p) * (scale.sqrt() * z.sqrt());
            a = d + z;
            d = d - (bcmax / z) * bcmis;
            let tau = c.hypot(z);
            (cs, sn) = (z / tau, c / tau);
            b = b - c;
            c = zero;
        } else {
            // Complex eigenvalues, or real ones almost equal: make the diagonal items equal.
            let mut sigma = b + c;
            let mut count = 0;
            loop {
                count += 1;
                let scale = temp.abs().max(sigma.abs());
                if scale >= safmx2 { (sigma, temp) = (sigma * safmn2, temp * safmn2); if count <= 20 { continue; } }
                if scale <= safmn2 { (sigma, temp) = (sigma * safmx2, temp * safmx2); if count <= 20 { continue; } }
                break;
            }
            let p = half * temp;
            let tau = sigma.hypot(temp);
            cs = (half * (one + sigma.abs() / tau)).sqrt();
            sn = -(p / (tau * cs)) * sign(sigma);
            // [aa bb; cc dd] = [a b; c d] [cs -sn; sn cs], then [a b; c d] = [cs sn; -sn cs] [aa bb; cc dd].
            let (aa, bb, cc, dd) = (a * cs + b * sn, -a * sn + b * cs, c * cs + d * sn, -c * sn + d * cs);
            (a, b, c, d) = (aa * cs + cc * sn, bb * cs + dd * sn, -aa * sn + cc * cs, -bb * sn + dd * cs);
            let temp = half * (a + d);
            (a, d) = (temp, temp);
            if c != zero {
                if b == zero {
                    (b, c) = (-c, zero);
                    (cs, sn) = (-sn, cs);
                } else if sign(b) == sign(c) {
                    // Real eigenvalues: reduce to upper triangular form.
                    let (sab, sac) = (b.abs().sqrt(), c.abs().sqrt());
                    let p = sign(c) * (sab * sac);
                    let tau = one / (b + c).abs().sqrt();
                    (a, d) = (temp + p, temp - p);
                    (b, c) = (b - c, zero);
                    let (cs1, sn1) = (sab * tau, sac * tau);
                    (cs, sn) = (cs * cs1 - sn * sn1, cs * sn1 + sn * cs1);
                }
            }
        }
    }
    let im = if c == zero { zero } else { b.abs().sqrt() * c.abs().sqrt() };
    ([a, b, c, d], [a, im, d, -im], (cs, sn))
}

/// The eigenvectors of the quasi-triangular `n`×`n` Schur form `t`, whose eigenvalues are `values`, as the columns of an `n`×`n` matrix. A real eigenvalue's vector fills its own column. A complex pair's first column holds the real part of the vector for the eigenvalue with the positive imaginary part, and its second column the imaginary part. Each vector comes from back substitution in `t - λ I`, as LAPACK's `dtrevc` does it, with a divisor smaller than `ε` times `t`'s size raised to that. Complex quotients follow Smith's method, which never squares a divisor. A vector whose items grow past the square root of the largest float scales down, so that sums of its products never overflow.
fn quasi_vectors<T: Native>(t: &[T], ld: usize, n: usize, values: &[Complex<T>]) -> Vec<T> {
    let zero = T::zero();
    let size = (0..n).fold(zero, |s, i| t[i * ld + i.saturating_sub(1)..i * ld + n].iter().fold(s, |s, x| s + x.abs()));
    let smin = (T::epsilon() * size).max(T::min_positive_value() * T::narrow(n as f64) / T::epsilon());
    let bound = T::max_value().sqrt();
    let mut x = vec![zero; n * n];
    let (mut xr, mut xi) = (vec![zero; n], vec![zero; n]);
    let mut k = n;
    while k > 0 {
        k -= 1;
        // A complex pair's second eigenvalue, with the negative imaginary part, comes first on the way up.
        let pair = values[k].im != zero;
        let top = if pair { k - 1 } else { k };
        xr[..=k].fill(zero);
        xi[..=k].fill(zero);
        let lambda = values[top];
        if pair {
            let (b, c) = (t[top * ld + k], t[k * ld + top]);
            if b.abs() >= c.abs() { (xr[top], xi[k]) = (T::one(), values[top].im / b) } else { (xr[top], xi[k]) = (-values[top].im / c, T::one()) }
        } else { xr[k] = T::one(); }
        // Rows i to k are solved. Each step solves the row above them, or the 2×2 block that ends there.
        let mut i = top;
        while i > 0 {
            let at = |p: usize, q: usize| Complex::new(t[p * ld + q], zero);
            let raised = |d: Complex<T>| if d.norm() < smin { Complex::new(smin, zero) } else { d };
            // A real eigenvalue's vector has no imaginary part to sum.
            let r = |p: usize| {
                let row = &t[p * ld + i..p * ld + k + 1];
                Complex::new(-dot(row, &xr[i..=k]), if pair { -dot(row, &xi[i..=k]) } else { zero })
            };
            let block = i >= 2 && t[(i - 1) * ld + i - 2] != zero;
            let (p, rows) = if block { (i - 2, 2) } else { (i - 1, 1) };
            let solved = if block {
                let q = i - 1;
                let (a, b, c, d) = (at(p, p) - lambda, at(p, q), at(q, p), at(q, q) - lambda);
                let (rp, rq, det) = (r(p), r(q), raised(a * d - b * c));
                [(rp * d - b * rq).quotient(det), (a * rq - c * rp).quotient(det)]
            } else { [r(p).quotient(raised(at(p, p) - lambda)), Complex::new(zero, zero)] };
            for (row, x) in (p..p + rows).zip(solved) { (xr[row], xi[row]) = (x.re, x.im) }
            i = p;
            let largest = solved[..rows].iter().fold(zero, |m, x| m.max(x.re.abs()).max(x.im.abs()));
            if largest > bound { let shrink = largest.recip(); for x in xr[i..=k].iter_mut().chain(&mut xi[i..=k]) { *x = *x * shrink } }
        }
        for row in 0..=k { x[row * n + top] = xr[row]; if pair { x[row * n + k] = xi[row] } }
        k = top;
    }
    x
}

/// The eigenvectors that the columns of the real `n`×`n` matrix `v` give, as `quasi_vectors` lays them out for the eigenvalues `values`, each of unit length.
fn real_units<T: Native>(v: &[T], n: usize, values: &[Complex<T>]) -> Vec<Complex<T>> {
    let zero = T::zero();
    let mut vectors = vec![Complex::new(zero, zero); n * n];
    let mut j = 0;
    while j < n {
        let pair = values[j].im != zero;
        let z: Vec<Complex<T>> = (0..n).map(|i| Complex::new(v[i * n + j], if pair { v[i * n + j + 1] } else { zero })).collect();
        let scale = unit(&z);
        for (i, &x) in z.iter().enumerate() { vectors[i * n + j] = x * scale; if pair { vectors[i * n + j + 1] = (x * scale).conj() } }
        j += if pair { 2 } else { 1 };
    }
    vectors
}

/// The factor that scales the vector `z` to unit length. Dividing by the largest modulus before squaring keeps the sum of squares finite.
fn unit<R: Native>(z: &[Complex<R>]) -> R {
    let largest = z.iter().fold(R::zero(), |m, x| m.max(x.norm()));
    (largest * z.iter().fold(R::zero(), |s, &x| s + (x / largest).norm_sqr()).sqrt()).recip()
}

/// Reduces the complex upper Hessenberg matrix `h`, whose rows start `ld` items apart, to upper triangular Schur form by single-shift QR iterations, as LAPACK's `zlahqr` does, and gives the eigenvalues from its diagonal. `zt` and `wantt` act as in `francis`. Unit scalings of rows and columns first make each subdiagonal item real. Each iteration keeps them real. `None` where an eigenvalue takes more than `30 max(10, n)` iterations.
fn single_shift<R: Native>(h: &mut [Complex<R>], ld: usize, zt: &mut [Complex<R>], n: usize, wantt: bool) -> Option<Vec<Complex<R>>> {
    // Applies the similarity `D⁻¹ h D`, where `D` is the identity with the unit factor `d` at position `j`. Row `j`'s items in `columns` scale by `d̄`. Column `j`'s items in `rows` scale by `d`, as does row `j` of Zᵀ where `zt` isn't empty.
    fn rescale<R: Native>(
        h: &mut [Complex<R>], ld: usize, zt: &mut [Complex<R>], n: usize, j: usize, d: Complex<R>, columns: Range<usize>, rows: Range<usize>) {
        for c in columns { h[j * ld + c] = h[j * ld + c] * d.conj() }
        for r in rows { h[r * ld + j] = h[r * ld + j] * d }
        if let Some(row) = zt.get_mut(j * n..(j + 1) * n) { for x in row { *x = *x * d } }
    }
    let (rzero, rone) = (R::zero(), R::one());
    let zero = Complex::new(rzero, rzero);
    let mut w = vec![zero; n];
    if n == 0 { return Some(w); }
    let (ulp, safmin) = (R::epsilon(), R::min_positive_value());
    let smlnum = safmin * (R::narrow(n as f64) / ulp);
    let (dat1, half) = (R::narrow(0.75), R::narrow(0.5));
    let itmax = 30 * n.max(10);
    for i in 1..n {
        let s = h[i * ld + i - 1];
        if s.im != rzero {
            // Dividing by `abs1` first keeps the modulus from underflowing.
            let sc = s.div_real(s.abs1());
            h[i * ld + i - 1] = Complex::new(s.norm(), rzero);
            rescale(h, ld, zt, n, i, sc.div_real(sc.norm()), i..n, 0..(i + 2).min(n));
        }
    }
    let (mut i1, mut i2) = (0, n - 1);
    // Iterations since the last deflation. Every tenth one takes an exceptional shift.
    let mut kdefl = 0;
    let mut i = n - 1;
    loop {
        // Rows and columns l to i form the active block. Eigenvalues after i have converged.
        let mut l = 0;
        let mut its = 0;
        loop {
            // The last negligible subdiagonal item at or above row i splits off the block that holds row i.
            let mut k = i;
            while k > l {
                let sub = h[k * ld + k - 1];
                if sub.abs1() <= smlnum { break; }
                let mut tst = h[(k - 1) * ld + k - 1].abs1() + h[k * ld + k].abs1();
                if tst == rzero { if k >= 2 { tst = tst + h[(k - 1) * ld + k - 2].re.abs() } if k + 1 < n { tst = tst + h[(k + 1) * ld + k].re.abs() } }
                // Ahues and Tisseur's conservative test, as LAPACK applies it.
                if sub.re.abs() <= ulp * tst {
                    let (sub, sup) = (sub.abs1(), h[(k - 1) * ld + k].abs1());
                    let (ab, ba) = (sub.max(sup), sub.min(sup));
                    let (d, e) = (h[k * ld + k].abs1(), (h[(k - 1) * ld + k - 1] - h[k * ld + k]).abs1());
                    let (aa, bb) = (d.max(e), d.min(e));
                    let s = aa + ab;
                    if ba * (ab / s) <= smlnum.max(ulp * (bb * (aa / s))) { break; }
                }
                k -= 1;
            }
            l = k;
            if l > 0 { h[l * ld + l - 1] = zero }
            if l >= i { break; }
            its += 1;
            if its > itmax { return None; }
            kdefl += 1;
            if !wantt { (i1, i2) = (l, i) }
            let shift = if kdefl % 20 == 0 { h[i * ld + i] + dat1 * h[i * ld + i - 1].re.abs() }
            else if kdefl % 10 == 0 { h[l * ld + l] + dat1 * h[(l + 1) * ld + l].re.abs() } else {
                // Wilkinson's shift: the eigenvalue of the trailing 2×2 block nearer h[i][i].
                let t = h[i * ld + i];
                let u = h[(i - 1) * ld + i].sqrt() * h[i * ld + i - 1].sqrt();
                let s = u.abs1();
                if s == rzero { t } else {
                    let x = (h[(i - 1) * ld + i - 1] - t) * half;
                    let sx = x.abs1();
                    let s = s.max(sx);
                    let (xs, us) = (x.div_real(s), u.div_real(s));
                    let mut y = (xs * xs + us * us).sqrt() * s;
                    if sx > rzero { let unit = x.div_real(sx); if unit.re * y.re + unit.im * y.im < rzero { y = -y } }
                    t - u * u.quotient(x + y)
                }
            };
            // The sweep starts at the lowest row m where a small subdiagonal item lets it begin without disturbing the rows above. v is the first column of the shifted matrix there, scaled.
            let mut m = i - 1;
            let mut v;
            loop {
                let (h11, h22) = (h[m * ld + m], h[(m + 1) * ld + m + 1]);
                let h11s = h11 - shift;
                let h21 = h[(m + 1) * ld + m].re;
                let s = h11s.abs1() + h21.abs();
                v = [h11s.div_real(s), Complex::new(h21 / s, rzero)];
                if m == l { break; }
                if h[m * ld + m - 1].re.abs() * v[1].re.abs() <= ulp * (v[0].abs1() * (h11.abs1() + h22.abs1())) { break; }
                m -= 1;
            }
            // The single-shift sweep: a reflection of order 2 makes a bulge, and each later one chases it down a row. v[1] is real before each reflection, so t2 is real.
            for k in m..i {
                if k > m { v = [h[k * ld + k - 1], h[(k + 1) * ld + k - 1]] }
                let t1 = reflector(&mut v);
                if k > m {
                    h[k * ld + k - 1] = v[0];
                    h[(k + 1) * ld + k - 1] = zero;
                }
                let (v2, t2) = (v[1], (t1 * v[1]).re);
                let (x, y) = two_rows(h, ld, k, k + 1);
                for (a, b) in x[k..=i2].iter_mut().zip(&mut y[k..=i2]) {
                    let sum = t1.conj() * *a + *b * t2;
                    (*a, *b) = (*a - sum, *b - sum * v2);
                }
                for row in h[i1 * ld..((k + 2).min(i) + 1) * ld].chunks_exact_mut(ld) {
                    let sum = t1 * row[k] + row[k + 1] * t2;
                    (row[k], row[k + 1]) = (row[k] - sum, row[k + 1] - sum * v2.conj());
                }
                if wantt {
                    let (x, y) = two_rows(zt, n, k, k + 1);
                    for (a, b) in x.iter_mut().zip(y.iter_mut()) {
                        let sum = t1 * *a + *b * t2;
                        (*a, *b) = (*a - sum, *b - sum * v2.conj());
                    }
                }
                if k == m && m > l {
                    // A sweep that starts below row l scales the rows and columns it starts in, so that h[m][m - 1] stays real.
                    let temp = Complex::new(rone, rzero) - t1;
                    let temp = temp.div_real(temp.norm());
                    h[(m + 1) * ld + m] = h[(m + 1) * ld + m] * temp.conj();
                    if m + 2 <= i { h[(m + 2) * ld + m + 1] = h[(m + 2) * ld + m + 1] * temp }
                    for j in (m..=i).filter(|&j| j != m + 1) { rescale(h, ld, zt, n, j, temp.conj(), j + 1..i2 + 1, i1..j) }
                }
            }
            let s = h[i * ld + i - 1];
            if s.im != rzero {
                let r = s.norm();
                h[i * ld + i - 1] = Complex::new(r, rzero);
                rescale(h, ld, zt, n, i, s.div_real(r), i + 1..i2 + 1, i1..i);
            }
        }
        w[i] = h[i * ld + i];
        kdefl = 0;
        if i == 0 { return Some(w); }
        i -= 1;
    }
}

/// The eigenvectors of the upper triangular `n`×`n` Schur form `t`, whose rows start `ld` items apart, as the columns of an `n`×`n` matrix. Column `k` solves `(t - t_kk I) x = 0`, with `x_k = 1` and zeros below it, by back substitution, as LAPACK's `ztrevc` does. Small divisors raise to `smin`, and large vectors scale down, as in `quasi_vectors`.
fn triangular_vectors<R: Native>(t: &[Complex<R>], ld: usize, n: usize) -> Vec<Complex<R>> {
    let rzero = R::zero();
    let size = (0..n).fold(rzero, |s, i| t[i * ld + i..i * ld + n].iter().fold(s, |s, x| s + x.norm()));
    let smin = (R::epsilon() * size).max(R::min_positive_value() * R::narrow(n as f64) / R::epsilon());
    let bound = R::max_value().sqrt();
    let mut x = vec![Complex::new(rzero, rzero); n * n];
    let mut col = x[..n].to_vec();
    for k in (0..n).rev() {
        let lambda = t[k * ld + k];
        col[k] = Complex::new(R::one(), rzero);
        for p in (0..k).rev() {
            let d = t[p * ld + p] - lambda;
            let d = if d.norm() < smin { Complex::new(smin, rzero) } else { d };
            col[p] = (-dot(&t[p * ld + p + 1..p * ld + k + 1], &col[p + 1..=k])).quotient(d);
            let largest = col[p].re.abs().max(col[p].im.abs());
            if largest > bound { let shrink = largest.recip(); for x in &mut col[p..=k] { *x = *x * shrink } }
        }
        for row in 0..=k { x[row * n + k] = col[row] }
    }
    x
}
