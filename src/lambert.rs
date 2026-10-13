//! The principal branch W0 of the Lambert W function: the w with w e^w = x. Near the branch point x = -1/e, W0 + 1 is Corless et
//! al.'s series in p = √(2(ex + 1)). Elsewhere two steps of Fritsch's iteration refine a first estimate. The complex version runs
//! Halley steps from first estimates chosen by region, as SciPy's `lambertw` does.
use crate::{
    arc,
    complex as cx,
    dd::Double,
    float::{horner, Native},
    ln::{ln, log1p},
    pow as p, LambertW, FUSED,
};
use num_complex::Complex;

/// e as a double word.
const E: (f64, f64) = (2.718281828459045, 1.4456468917292502e-16);
/// The coefficients μ_36 down to μ_1 of W0 + 1 as a series in p, from Corless et al.'s recurrence. Below `NEAR` their sum is W0 + 1 to
/// within an ulp of W0.
const BRANCH: [f64; 36] = [
    -1.240275440042247e-07, 1.8054472775101644e-07, -2.630378619271863e-07, 3.835663851491814e-07, -5.598551881378796e-07,
    8.179940565280035e-07, -1.1964453089157258e-06, 1.7520067268263412e-06, -2.5687050390550955e-06, 3.7710949611072535e-06,
    -5.544203208567359e-06, 8.163531982496613e-06, -1.2040352739559977e-05, 1.7790345805079586e-05, -2.63380647472311e-05,
    3.907668486743905e-05, -5.811360750441382e-05, 8.665035805208128e-05, -0.00012957426685274883, 0.00019438727605453933,
    -0.00029267722472962746, 0.0004424730618146209, -0.0006720616311561362, 0.0010262633205076071, -0.0015769303446867841,
    0.0024408779911439826, -0.0038112980348919993, 0.006014543252956118, -0.009616892024299432, 0.01563563253233392,
    -0.02598471487360376, 0.044502314814814814, -0.07962962962962963, 0.1527777777777778, -0.3333333333333333, 1.0,
];
/// The p below which W0 comes from `BRANCH` alone. A Fritsch step's error grows as 1 + W0 nears 0, in proportion to |W0| / (1 + W0).
const NEAR: f64 = 0.5;

/// W0(x), computed in f64. x below -1/e gives NaN. The float nearest -1/e lies below it at f32 and f64, so an x below -1/e by less
/// than half an ulp of x gives -1.
#[inline(always)]
pub(crate) fn lambertw<R: Native, const FUSED: bool>(x: R) -> R {
    let half_ulp = 0.5 * R::epsilon().wide();
    let x = x.wide();
    // q = ex + 1 to within an ulp of q, through ex as a double word: near the branch point ex + 1 cancels.
    let ex = Double::new(E.0, E.1).mul_float::<FUSED>(x);
    let q = (ex.hi + 1.0) + ex.lo;
    let q = if (q < 0.0) & (q >= -half_ulp) { 0.0 } else { q };
    let p = (2.0 * q).sqrt();
    let near = p * horner::<f64, FUSED, 36>(p, BRANCH) - 1.0;
    // The first estimate: the start of the branch series up to p = 1, which covers x up to about -0.18, and Winitzki's formula above.
    let [.., m4, m3, m2, m1] = BRANCH;
    let l = log1p::<f64, FUSED>(x);
    let w = if p < 1.0 { p * horner::<f64, FUSED, 4>(p, [m4, m3, m2, m1]) - 1.0 } else { l * (1.0 - log1p::<f64, FUSED>(l) / (2.0 + l)) };
    // Two steps of Fritsch's iteration, each about fourth order. The last step's error is about ln's error times |w| / (1 + w).
    let fritsch = |w: f64| {
        let z = ln::<f64, FUSED>(x / w) - w;
        let t = 1.0 + w;
        let s = 2.0 * t * (t + z * (2.0 / 3.0));
        w * (1.0 + z / t * (s - z) / (s - 2.0 * z))
    };
    let w = fritsch(fritsch(w));
    let w = if p < NEAR { near } else { w };
    // 0 and ∞ give themselves. The steps divide 0 by 0 at 0, and Winitzki's formula is NaN at ∞.
    let w = if (x == 0.0) | (x == f64::INFINITY) { x } else { w };
    R::narrow(if q < 0.0 { f64::NAN } else { w })
}

/// The most Halley steps that complex W0 takes. From its first estimates it needs at most 8.
const STEPS: usize = 32;

/// W0(z) for a complex z, computed in f64. The branch cut runs along the real axis below -1/e. z on the cut takes the side that the
/// sign of its imaginary part gives, as `ln` does. An infinite z gives ∞ at z's angle, and a NaN part gives NaN.
pub(crate) fn complex<R: Native>(z: Complex<R>) -> Complex<R> {
    let w = complex_wide(Complex::new(z.re.wide(), z.im.wide()));
    Complex::new(R::narrow(w.re), R::narrow(w.im))
}

fn complex_wide(z: Complex<f64>) -> Complex<f64> {
    let norm = |v: Complex<f64>| p::hypot::<f64, FUSED>(v.re, v.im);
    if z.re.is_nan() || z.im.is_nan() { return Complex::new(f64::NAN, f64::NAN) }
    if z.re.is_infinite() || z.im.is_infinite() { return Complex::new(f64::INFINITY, arc::atan2::<f64, FUSED>(z.im, z.re)) }
    if z.re == 0.0 && z.im == 0.0 { return z }
    // q = ez + 1, whose real part cancels near the branch point, as `lambertw`'s does.
    let ex = Double::new(E.0, E.1).mul_float::<FUSED>(z.re);
    let p = cx::sqrt::<f64, FUSED>(Complex::new((ex.hi + 1.0) + ex.lo, E.0 * z.im) * 2.0);
    if norm(p) < NEAR { return p * BRANCH.iter().fold(Complex::new(0.0, 0.0), |s, &c| s * p + c) - 1.0 }
    let mut w = if norm(z + (1.0 / std::f64::consts::E)) < 0.3 {
        let [.., m4, m3, m2, m1] = BRANCH;
        p * (((p * m4 + m3) * p + m2) * p + m1) - 1.0
    } else if -1.0 < z.re && z.re < 1.5 && z.im.abs() < 1.0 && -2.5 * z.im.abs() - 0.2 < z.re {
        // The [2/2] Padé approximant of W0(z)/z at 0.
        z * ((z * (17.0 / 60.0) + 1.9) * z + 1.0) / ((z * (101.0 / 60.0) + 2.9) * z + 1.0)
    } else {
        let l1 = cx::ln::<f64, FUSED>(z);
        let l2 = cx::ln::<f64, FUSED>(l1);
        l1 - l2 + l2 / l1
    };
    let mut last = f64::INFINITY;
    for _ in 0..STEPS {
        // The Halley step on w e^w - z, divided through by e^w: the step's parts stay near |w|, where a complex division of terms
        // near |z| would square them and overflow.
        let g = w - z * cx::exp::<f64, FUSED>(-w);
        let d = g / ((w + 1.0) - (w + 2.0) * g / (w * 2.0 + 2.0));
        // Near the last bit, rounding stops the steps getting smaller.
        if norm(d) >= last { break }
        w -= d;
        last = norm(d);
        if last <= 0.5 * f64::EPSILON * norm(w) { break }
    }
    w
}

forms!(LambertW at Complex<f32>, Complex<f64>: |_, [z]| complex(z));
