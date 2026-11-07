use super::*;
use crate::FUSED;

pub(super) fn hyperbolic_parts<R: Native + Cast>(x: &[R], y: &[R], cosh: bool, re: &mut Vec<R>, im: &mut Vec<R>) {
    let (mut sh, mut ch, mut s, mut c) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    real(Sinh, [x], &mut sh);
    real(Cosh, [x], &mut ch);
    sin_cos(y, &mut s, &mut c);
    let zero = R::zero();
    if cosh {
        re.extend(ch.iter().zip(&c).map(|(&ch, &c)| ch * c));
        im.extend(x.iter().zip(y).zip(sh.iter().zip(&s)).map(|((&x, &y), (&sh, &s))| {
            if (x == zero) | (y == zero) { zero.copysign(x) * zero.copysign(y) } else { sh * s }
        }));
    } else {
        re.extend(x.iter().zip(&sh).zip(&c).map(|((&x, &sh), &c)| if x == zero { x } else { sh * c }));
        im.extend(y.iter().zip(&ch).zip(&s).map(|((&y, &ch), &s)| if y == zero { y } else { ch * s }));
    }
}

pub(super) fn moduli<R: Native + Cast>(x: &[R], y: &[R], out: &mut Vec<R>) {
    if R::WIDE { real(Hypot, [x, y], out); } else { out.extend(x.iter().zip(y).map(|(&x, &y)| modulus::<R, FUSED>(x, y))) }
}

pub(super) fn sqrt_parts<R: Native + Cast>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) {
    let mut h = Vec::new();
    moduli(x, y, &mut h);
    unsafe { crate::slice::appended(re, x.len(), |re| {
        crate::slice::appended(im, x.len(), |im| {
            for ((((&x, &y), &h), re), im) in x.iter().zip(y).zip(&h).zip(re).zip(im) {
                let (r, i) = root::<R, FUSED>(x, y, h);
                re.write(r);
                im.write(i);
            }
        })
    }) }
}

pub(super) fn rough_root<R: Native + Cast>(z: Complex<R>) -> bool {
    (scaled(z).1 != R::zero()) | z.im.is_infinite() | ((z.re == R::zero()) & (z.im == R::zero()))
}

fn root_pair<R: Native + Cast>([ax, ay]: [&[R]; 2], [bx, by]: [&[R]; 2]) -> [(Vec<R>, Vec<R>); 2] {
    let mut roots = [(Vec::new(), Vec::new()), (Vec::new(), Vec::new())];
    let [a, b] = &mut roots;
    sqrt_parts(ax, ay, &mut a.0, &mut a.1);
    sqrt_parts(bx, by, &mut b.0, &mut b.1);
    roots
}

pub(super) fn roots_parts<R: Native + Cast>(x: &[R], y: &[R]) -> [(Vec<R>, Vec<R>); 2] {
    let one = R::one();
    let below: Vec<R> = x.iter().map(|&x| one - x).collect();
    let minus_y: Vec<R> = y.iter().map(|&y| -y).collect();
    let above: Vec<R> = x.iter().map(|&x| one + x).collect();
    root_pair([&below, &minus_y], [&above, y])
}

pub(super) fn rough_roots<R: Native + Cast>(z: Complex<R>) -> bool { rough_root(c(R::one() - z.re, -z.im)) | rough_root(c(R::one() + z.re, z.im)) }

pub(super) fn shifted_roots_parts<R: Native + Cast>(x: &[R], y: &[R]) -> [(Vec<R>, Vec<R>); 2] {
    let one = R::one();
    let below: Vec<R> = x.iter().map(|&x| x - one).collect();
    let above: Vec<R> = x.iter().map(|&x| x + one).collect();
    root_pair([&below, y], [&above, y])
}

pub(super) fn rough_shifted_roots<R: Native + Cast>(z: Complex<R>) -> bool { rough_root(c(z.re - R::one(), z.im)) | rough_root(c(z.re + R::one(), z.im)) }

fn times_parts<R: Native + Cast>((ar, ai): &(Vec<R>, Vec<R>), (br, bi): &(Vec<R>, Vec<R>), re: &mut Vec<R>, im: &mut Vec<R>) {
    re.extend(ar.iter().zip(ai).zip(br.iter().zip(bi)).map(|((&ar, &ai), (&br, &bi))| ar * br - ai * bi));
    im.extend(ar.iter().zip(ai).zip(br.iter().zip(bi)).map(|((&ar, &ai), (&br, &bi))| ar * bi + ai * br));
}

pub(super) fn root_products_parts<R: Native + Cast>(
    roots: impl Fn(&[R], &[R]) -> [(Vec<R>, Vec<R>); 2], negate: bool, x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) {
    let [a, b] = roots(x, y);
    let start = re.len();
    times_parts(&a, &b, re, im);
    let sign = if negate { -R::one() } else { R::one() };
    for (((re, im), &x), &y) in re[start..].iter_mut().zip(&mut im[start..]).zip(x).zip(y) {
        let r = balanced(c(*re, *im), sign * x, y);
        (*re, *im) = (r.re, r.im);
    }
}

pub(super) fn sqrt1mx2_parts<R: Native + Cast>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) { root_products_parts(roots_parts, true, x, y, re, im) }

pub(super) fn turned_parts<R: Native + Cast>(x: &[R], y: &[R], g: impl Fn(&[R], &[R], &mut Vec<R>, &mut Vec<R>), re: &mut Vec<R>, im: &mut Vec<R>) {
    let minus_y: Vec<R> = y.iter().map(|&v| -v).collect();
    g(&minus_y, x, re, im);
}

pub(super) fn unturned_parts<R: Native + Cast>(x: &[R], y: &[R], g: impl Fn(&[R], &[R], &mut Vec<R>, &mut Vec<R>), re: &mut Vec<R>, im: &mut Vec<R>) {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    turned_parts(x, y, g, &mut a, &mut b);
    re.extend(b);
    im.extend(a.iter().map(|&v| -v));
}

pub(super) fn sin_parts<R: Native + Cast>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) {
    unturned_parts(x, y, |x, y, re, im| hyperbolic_parts(x, y, false, re, im), re, im)
}

pub(super) fn tan_parts<R: Native + Cast>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) { unturned_parts(x, y, tanh_parts, re, im) }

pub(super) fn asin_parts<R: Native + Cast>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) {
    let [(ar, ai), (br, bi)] = roots_parts(x, y);
    let d: Vec<R> = ar.iter().zip(&ai).zip(br.iter().zip(&bi)).map(|((&ar, &ai), (&br, &bi))| ar.mla::<FUSED>(br, -(ai * bi))).collect();
    let s: Vec<R> = ar.iter().zip(&ai).zip(br.iter().zip(&bi)).map(|((&ar, &ai), (&br, &bi))| ar.mla::<FUSED>(bi, -(ai * br))).collect();
    real(Atan2, [x, &d], re);
    real(Asinh, [&s], im);
}

pub(super) fn acos_parts<R: Native + Cast>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) {
    let mut half = Vec::new();
    let [(ar, ai), (br, bi)] = roots_parts(x, y);
    let s: Vec<R> = br.iter().zip(&bi).zip(ar.iter().zip(&ai)).map(|((&br, &bi), (&ar, &ai))| br.mla::<FUSED>(ai, -(bi * ar))).collect();
    real(Atan2, [&ar, &br], &mut half);
    re.extend(half.iter().map(|&h| h + h));
    real(Asinh, [&s], im);
}

pub(super) fn acosh_parts<R: Native + Cast>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) {
    let mut half = Vec::new();
    let [(ar, ai), (br, bi)] = shifted_roots_parts(x, y);
    let s: Vec<R> = ar.iter().zip(&ai).zip(br.iter().zip(&bi)).map(|((&ar, &ai), (&br, &bi))| ar.mla::<FUSED>(br, ai * bi)).collect();
    real(Asinh, [&s], re);
    real(Atan2, [&ai, &br], &mut half);
    im.extend(half.iter().map(|&h| h + h));
}

pub(super) fn tanh_parts<R: Native + Cast>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) {
    let (one, mut sy, mut cy, mut s) = (R::one(), Vec::new(), Vec::new(), Vec::new());
    sin_cos(y, &mut sy, &mut cy);
    real(Sinh, [x], &mut s);
    let t: Vec<R> = sy.iter().zip(&cy).map(|(&sy, &cy)| sy / cy).collect();
    let b: Vec<R> = t.iter().map(|&t| t.mla::<FUSED>(t, one)).collect();
    let d: Vec<R> = b.iter().zip(&s).map(|(&b, &s)| (b * s).mla::<FUSED>(s, one)).collect();
    re.extend(b.iter().zip(&s).zip(&d).map(|((&b, &s), &d)| b * s * s.mla::<FUSED>(s, one).sqrt() / d));
    im.extend(t.iter().zip(&d).map(|(&t, &d)| t / d));
}

pub(super) fn rough_tanh<R: Native + Cast>(z: Complex<R>) -> bool { !(z.re.abs() <= tanh_limit::<R>()) | !z.im.is_finite() }

pub(super) fn atanh_parts<R: Native + Cast>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) {
    let (one, mut lr, mut li) = (R::one(), Vec::new(), Vec::new());
    let s: Vec<R> = x.iter().map(|&x| one.copysign(x)).collect();
    let ax: Vec<R> = x.iter().map(|&x| x.abs()).collect();
    let ay: Vec<R> = y.iter().zip(&s).map(|(&y, &s)| y * s).collect();
    let q: Vec<R> = ax.iter().zip(&ay).map(|(&x, &y)| R::narrow(4.0) * x / (one - x).mla::<FUSED>(one - x, y * y)).collect();
    let u: Vec<R> = ay.iter().map(|&y| y + y).collect();
    let v: Vec<R> = ax.iter().zip(&ay).map(|(&x, &y)| (one - x).mla::<FUSED>(one + x, -(y * y))).collect();
    real(Log1p, [&q], &mut lr);
    real(Atan2, [&u, &v], &mut li);
    re.extend(lr.iter().zip(&s).map(|(&r, &s)| r * R::narrow(0.25) * s));
    im.extend(li.iter().zip(&s).map(|(&i, &s)| i * R::narrow(0.5) * s));
}

pub(super) fn rough_atanh<R: Native + Cast>(z: Complex<R>) -> bool {
    !z.re.is_finite() | !z.im.is_finite() | (z.re.abs().max(z.im.abs()) > atanh_huge::<R>() * R::narrow(0.5))
}

