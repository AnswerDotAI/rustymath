//! Complex slice bodies split into real passes. Rare inputs use the same object's scalar body; the map owner checks only the final complex result.
use super::*;
use crate::{Asinh, Atan2, Cos, Cosh, Exp, FUSED, Hypot, Ln, Log1p, Sin, Sinh, buffer::Cast, facts::Formula, recipe::NativeOps, slice::{self, Arg}};
use std::mem::MaybeUninit;

mod trig;
use trig::*;

fn real<K: Formula<N>, R: Native + Cast, const N: usize>(k: K, args: [&[R]; N], out: &mut Vec<R>) {
    let n = args[0].len();
    slice::appended(out, n, |out| crate::operation::real_slice(&k, &NativeOps::DEFAULT, args.map(Arg::Many), n, out));
}

fn sin_cos<R: Native + Cast>(y: &[R], sin: &mut Vec<R>, cos: &mut Vec<R>) {
    real(Sin, [y], sin);
    real(Cos, [y], cos);
}

fn split<K: Formula<N>, R: Native + Cast, const N: usize>(
    k: &K, ops: &NativeOps<{ FUSED }>, args: [Arg<'_, Complex<R>>; N], n: usize, out: &mut [MaybeUninit<Complex<R>>],
    fast: impl FnOnce([(&[R], &[R]); N], &mut Vec<R>, &mut Vec<R>), rare: impl Fn([Complex<R>; N], Complex<R>) -> bool) {
    let (mut parts, mut re, mut im) = ([(); N].map(|()| (Vec::new(), Vec::new())), Vec::new(), Vec::new());
    for ((xs, ys), a) in parts.iter_mut().zip(args) {
        match a {
            Arg::One(z) => {
                xs.resize(n, z.re);
                ys.resize(n, z.im);
            }
            Arg::Many(s) => {
                xs.extend(s[..n].iter().map(|z| z.re));
                ys.extend(s[..n].iter().map(|z| z.im));
            }
        }
    }
    fast(std::array::from_fn(|i| (&parts[i].0[..], &parts[i].1[..])), &mut re, &mut im);
    assert!(re.len() == n && im.len() == n);
    for ((slot, re), im) in out[..n].iter_mut().zip(re).zip(im) { slot.write(c(re, im)); }
    let values = slice::written(&mut out[..n]);
    for (i, value) in values.iter_mut().enumerate() {
        let x = std::array::from_fn(|j| args[j].nth(i));
        if rare(x, *value) { *value = k.complex(ops, x).expect("a complex slice has a scalar body"); }
    }
}

fn exp_parts<R: Native + Cast>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) {
    let (mut e, mut sy, mut cy) = (Vec::new(), Vec::new(), Vec::new());
    real(Exp, [x], &mut e);
    sin_cos(y, &mut sy, &mut cy);
    for ((&y, sy), cy) in y.iter().zip(&mut sy).zip(&mut cy) {
        if let Some((c, s)) = quarter(y) {
            *cy = c;
            *sy = s;
        }
    }
    re.extend(x.iter().zip(e.iter().zip(&cy)).map(|(&x, (&e, &c))| part(x, c, e)));
    im.extend(x.iter().zip(y).zip(e.iter().zip(&sy)).map(|((&x, &y), (&e, &s))| if y == R::zero() { y } else { part(x, s, e) }));
}

fn ln_parts<R: Native + Cast>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) {
    let (mut h, mut lh, mut lt) = (Vec::new(), Vec::new(), Vec::new());
    real(Hypot, [x, y], &mut h);
    real(Ln, [&h], &mut lh);
    let one = R::one();
    let t: Vec<R> = x.iter().zip(y).map(|(&x, &y)| (x - one).mla::<FUSED>(x + one, y * y)).collect();
    real(Log1p, [&t], &mut lt);
    re.extend(h.iter().zip(lh).zip(lt).map(|((&h, lh), lt)| if h > R::narrow(0.5) && h < R::narrow(2.0) { lt * R::narrow(0.5) } else { lh }));
    real(Atan2, [y, x], im);
}

fn pow_parts<R: Native + Cast>([x, y]: [&[R]; 2], [u, v]: [&[R]; 2], re: &mut Vec<R>, im: &mut Vec<R>) {
    let (mut lr, mut li) = (Vec::new(), Vec::new());
    ln_parts(x, y, &mut lr, &mut li);
    let pr: Vec<R> = u.iter().zip(v).zip(lr.iter().zip(&li)).map(|((&u, &v), (&lr, &li))| u * lr - v * li).collect();
    let pi: Vec<R> = u.iter().zip(v).zip(lr.iter().zip(&li)).map(|((&u, &v), (&lr, &li))| u * li + v * lr).collect();
    exp_parts(&pr, &pi, re, im);
}

// The usual square-root block avoids scaling without changing scalar sqrt's result.
#[inline(always)]
fn plain_root<R: Native>(z: Complex<R>) -> bool {
    let (lo, hi) = if R::WIDE { (1.4916681462400413e-154, 6.703903964971299e153) } else { (9.860761315262648e-32, f32::MAX as f64) };
    let (a, b) = (z.re.abs(), z.im.abs());
    let part = |p: R| (p == R::zero()) | ((p >= R::narrow(lo)) & (p <= R::narrow(hi)));
    part(a) & part(b) & ((a != R::zero()) | (b != R::zero()))
}

pub(crate) fn sqrt<K: Formula<1>, R: Native + Cast>(
    k: &K, ops: &NativeOps<{ FUSED }>, [arg]: [Arg<'_, Complex<R>>; 1], n: usize, out: &mut [MaybeUninit<Complex<R>>]) where Complex<R>: slice::KernelItem {
    slice::each_usual(arg, n, out, (plain_root, #[inline(always)] |z| {
        let h = if R::WIDE { z.re.mla::<FUSED>(z.re, z.im * z.im).sqrt() } else { modulus::<R, FUSED>(z.re, z.im) };
        let (re, im) = root::<R, FUSED>(z.re, z.im, h);
        c(re, im)
    },), slice::none(), |z| k.complex(ops, [z]).unwrap());
}

macro_rules! unary {
    ($($name:ident: $parts:expr, $rare:expr;)+) => {$(
        pub(crate) fn $name<K: Formula<1>, R: Native + Cast>(k: &K, ops: &NativeOps<{ FUSED }>, args: [Arg<'_, Complex<R>>; 1], n: usize, out: &mut [MaybeUninit<Complex<R>>]) {
            split(k, ops, args, n, out, |[(x, y)], re, im| ($parts)(x, y, re, im), $rare);
        }
    )+};
}

unary! {
    exp: exp_parts, |_, r| !finite(r);
    cis: |x, y, re, im| turned_parts(x, y, exp_parts, re, im), |_, r| !finite(r);
    ln: ln_parts, |[z], r| (scaled(z).1 != R::zero()) | !finite(r);
    sinh: |x, y, re, im| hyperbolic_parts(x, y, false, re, im), |_, _| false;
    cosh: |x, y, re, im| hyperbolic_parts(x, y, true, re, im), |_, _| false;
    sin: sin_parts, |_, _| false;
    cos: |x, y, re, im| turned_parts(x, y, |x, y, re, im| hyperbolic_parts(x, y, true, re, im), re, im), |_, _| false;
    tan: tan_parts, |[z], _| rough_tanh(turned(z));
    tanh: tanh_parts, |[z], _| rough_tanh(z);
    asin: asin_parts, |[z], _| rough_roots(z);
    acos: acos_parts, |[z], _| rough_roots(z);
    asinh: |x, y, re, im| unturned_parts(x, y, asin_parts, re, im), |[z], _| rough_roots(turned(z));
    acosh: acosh_parts, |[z], _| rough_shifted_roots(z);
    atan: |x, y, re, im| unturned_parts(x, y, atanh_parts, re, im), |[z], _| rough_atanh(turned(z));
    atanh: atanh_parts, |[z], _| rough_atanh(z);
    sqrt1mx2: sqrt1mx2_parts, |[z], _| rough_roots(z);
    sqrt1px2: |x, y, re, im| turned_parts(x, y, sqrt1mx2_parts, re, im), |[z], _| rough_roots(turned(z));
    sqrtx2m1: |x, y, re, im| root_products_parts(shifted_roots_parts, false, x, y, re, im), |[z], _| rough_shifted_roots(z);
}

pub(crate) fn pow<K: Formula<2>, R: Native + Cast>(
    k: &K, ops: &NativeOps<{ FUSED }>, args: [Arg<'_, Complex<R>>; 2], n: usize, out: &mut [MaybeUninit<Complex<R>>]) {
    split(k, ops, args, n, out, |[(x, y), (u, v)], re, im| pow_parts([x, y], [u, v], re, im), |[z, w], r| {
        let (unit, zero, real) = pow_cases(z, w);
        unit | zero | real | (scaled(z).1 != R::zero()) | !finite(r)
    });
}
