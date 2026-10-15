//! Special functions: gamma, its logarithm and binomial coefficients, the error function and its inverses, and the regularized incomplete gamma and beta functions with their inverses. f64 computes in f64. Narrower types round an f64 result, from a cheaper f64 algorithm where one has the precision they need. `gamma`, `lgamma` and `erf` have no branch on their usual path, and loops that call them vectorize. The others choose a method by range, or iterate until they converge.
use crate::{Arg, Native, dd::Double, exp::{exp, exp_narrow, exp_plus}, float::{horner, sealed::Core}, ln::{ln, ln_double, ln_narrow, ln_split, log1p}};
use num_complex::Complex;
use std::{f64::consts::PI, mem::MaybeUninit};

/// Each function at every type: it computes in f64, and narrower types round the result.
macro_rules! in_f64 {
    ($($name:ident($($x:ident),+) = $core:ident;)+) => {$(
        #[inline(always)]
        pub(crate) fn $name<R: Native, const FUSED: bool>($($x: R),+) -> R { R::narrow($core::<FUSED>($($x.wide()),+)) }
    )+};
}

in_f64! {
    gammaincinv(a, p) = gammaincinv64;
    betainc(a, b, x) = betainc64;
    betaincinv(a, b, p) = betaincinv64;
}

/// Γ(x). f64 carries the Lanczos sum and exponent in double words. Narrower types compute in plain f64, whose 29 spare bits cover the exponent's amplification of its own rounding.
#[inline(always)]
pub(crate) fn gamma<R: Native, const FUSED: bool>(x: R) -> R { R::narrow(if R::WIDE { gamma64::<FUSED>(x.wide()) } else { gamma_plain::<FUSED>(x.wide()) }) }

/// Γ(x + 1).
#[inline(always)]
pub(crate) fn factorial<R: Native, const FUSED: bool>(x: R) -> R { gamma::<R, FUSED>(x + R::one()) }

/// ln |Γ(x)|, with f64 and narrower types computing as `gamma` does.
#[inline(always)]
pub(crate) fn lgamma<R: Native, const FUSED: bool>(x: R) -> R {
    R::narrow(if R::WIDE { lgamma64::<FUSED>(x.wide()) } else { lgamma_stirling::<FUSED>(x.wide()) })
}

/// The most factors that `binom` multiplies at f64 and at narrower types. C(n, k) with that many steps, the smaller of k and n - k, is at least C(2k, k), which overflows the type, so more steps give ±∞ at once.
const BINOM_STEPS: (f64, f64) = (600.0, 80.0);

/// C(n, k). For whole n and a whole k from 0, it multiplies by (top - i)/(i + 1) for i below the smaller of k and top - k, where top is n, or k - n - 1 for a negative n, which gives (-1)^k C(top, k). Each step's quotient is a whole number, so the result is exact while the result times top is a whole number the type holds. Other arguments take Γ(n + 1)/(Γ(k + 1) Γ(n - k + 1)) through the complex binomial coefficient's real part.
#[inline]
pub(crate) fn binom<R: Native, const FUSED: bool>(n: R, k: R) -> R {
    let whole = |v: R| v.trunc() == v;
    if whole(n) & whole(k) & (k >= R::zero()) {
        if (n >= R::zero()) & (k > n) { return R::zero(); }
        if k.is_finite() {
            let (top, negative) = if n >= R::zero() { (n, false) } else { (k - n - R::one(), (k * R::narrow(0.5)).fract() != R::zero()) };
            let steps = if k < top - k { k } else { top - k };
            let c = if steps > R::narrow(if R::WIDE { BINOM_STEPS.0 } else { BINOM_STEPS.1 }) { R::infinity() } else {
                let (mut c, mut i) = (R::one(), R::zero());
                while i < steps {
                    c = c * (top - i) / (i + R::one());
                    i = i + R::one();
                }
                c
            };
            return if negative { -c } else { c };
        }
    }
    crate::complex::binom::<R, FUSED>(Complex::new(n, R::zero()), Complex::new(k, R::zero())).re
}

/// The g of the Lanczos approximation that musl's `tgamma` uses, and g - 1/2.
const G: f64 = 6.024680040776729583740234375;
const G_HALF: f64 = 5.524680040776729583740234375;
/// e^-g.
const EXP_NEG_G: f64 = 0.00241832520904651;
/// 512 ln 2 as a double word, and 2^-512.
const LN2_512: (f64, f64) = (354.891356446692, 1.1873519686893054e-14);
const TWO_M512: f64 = 7.458340731200207e-155;
/// π - `PI`.
const PI_LO: f64 = 1.2246467991473532e-16;
/// ln(2π)/2, and the coefficients of Stirling's series after its leading terms, B_2k / (2k(2k-1)) for k from 10 down to 1, as a polynomial in 1/z², highest degree first: ln Γ(z) ≈ (z - 1/2) ln z - z + ln(2π)/2 + (1/z)(1/12 - 1/(360 z²) + 1/(1260 z⁴) - …). Narrower types need only the last three, and complex ln Γ all ten.
pub(crate) const HALF_LN_2PI: f64 = 0.9189385332046728;
pub(crate) const STIRLING: [f64; 10] = [
    -1.3924322169059011, 0.17964437236883057, -0.029550653594771242, 0.00641025641025641, -0.0019175269175269176, 0.0008417508417508417, -0.0005952380952380953,
    0.0007936507936507937, -0.002777777777777778, 0.08333333333333333,
];

/// The numerator and denominator of the Lanczos sum S from musl's `tgamma`, as polynomials in 1/y times y^12, highest degree first: Γ(y) = S(y) e^-g e^((y - 1/2)(ln(y + g - 1/2) - 1)).
const NUM: [f64; 13] = [
    23531376880.410759688572007674451636754734846804940, 42919803642.649098768957899047001988850926355848959,
    35711959237.355668049440185451547166705960488635843, 17921034426.037209699919755754458931112671403265390,
    6039542586.3520280050642916443072979210699388420708, 1439720407.3117216736632230727949123939715485786772,
    248874557.86205415651146038641322942321632125127801, 31426415.585400194380614231628318205362874684987640,
    2876370.6289353724412254090516208496135991145378768, 186056.26539522349504029498971604569928220784236328,
    8071.6720023658162106380029022722506138218516325024, 210.82427775157934587250973392071336271166969580291,
    2.5066282746310002701649081771338373386264310793408,
];
const DEN: [f64; 13] = [0.0, 39916800.0, 120543840.0, 150917976.0, 105258076.0, 45995730.0, 13339535.0, 2637558.0, 357423.0, 32670.0, 1925.0, 66.0, 1.0];
/// S(y) and the exponent, each as a double word, for y = `hi + lo` ≥ 1/2 with `lo` below half an ulp of `hi`. S is a ratio of polynomials in 1/y, whose positive terms cancel nowhere. They are double words too, because in floats Horner's rule loses about 3 ulps on them.
#[inline(always)]
fn lanczos<const FUSED: bool>(y: Double<f64>) -> (Double<f64>, Double<f64>) {
    let v = 1.0 / y.hi;
    (Double::horner::<FUSED>(v, NUM).div::<FUSED>(Double::horner::<FUSED>(v, DEN)), exponent::<FUSED>(y, true))
}

/// The Lanczos exponent (y - 1/2)(ln(y + g - 1/2) - 1) as a double word, for y as `lanczos` takes it, with the logarithm that `ln_dd` chooses by `fine`.
#[inline(always)]
fn exponent<const FUSED: bool>(y: Double<f64>, fine: bool) -> Double<f64> {
    let l = ln_dd::<FUSED>(Double::sum(y.hi, G_HALF).plus(Double::new(y.lo, 0.0)), fine);
    let m = l.plus(Double::new(-1.0, 0.0));
    let e = m.mul_float::<FUSED>(y.hi - 0.5);
    Double::new(e.hi, e.lo + m.hi * y.lo)
}

/// ln u for a double word u, as a double word, from `ln_double` if `fine` and from `ln_split` otherwise.
#[inline(always)]
fn ln_dd<const FUSED: bool>(u: Double<f64>, fine: bool) -> Double<f64> {
    let l = if fine { ln_double::<FUSED>(u.hi) } else { ln_split::<FUSED>(u.hi) };
    Double::new(l.hi, l.lo + u.lo / u.hi)
}

/// ln Γ(y) as a double word, for y > 0. From 1/2 up it keeps the Lanczos exponent's double word, so that a sum of such logarithms cancels without rounding. Below 1/2 it is small, and one float holds it.
fn lgamma_dd<const FUSED: bool>(y: Double<f64>) -> Double<f64> {
    if y.hi < 0.5 { return Double::new(lgamma64::<FUSED>(y.hi), 0.0); }
    let (s, e) = lanczos::<FUSED>(y);
    e.plus(Double::new(ln::<f64, FUSED>(s.hi * EXP_NEG_G), s.lo / s.hi))
}

/// sin πr / r and cos πr as polynomials in r², highest degree first: their Taylor series, which are within an ulp up to |r| = 1/4.
const SIN_PI: [f64; 9] = [
    7.952054001475513e-07, -2.1915353447830217e-05, 0.00046630280576761255, -0.0073704309457143504, 0.08214588661112823, -0.5992645293207921,
    2.5501640398773455, -5.16771278004997, 3.141592653589793,
];
const COS_PI: [f64; 10] = [
    -1.3878952462213771e-07, 4.303069587032947e-06, -0.0001046381049248457, 0.0019295743094039231, -0.02580689139001406, 0.2353306303588932,
    -1.3352627688545895, 4.0587121264167685, -4.934802200544679, 1.0,
];

/// sin πx. With n the nearest whole number to x, r = x - n is exact and |r| ≤ 1/2. sin π|r| comes from `SIN_PI` up to 1/4, and as cos π(1/2 - |r|) from `COS_PI` beyond, where 1/2 - |r| is exact too. No argument is rounded before the polynomials.
#[inline(always)]
fn sinpi<const FUSED: bool>(x: f64) -> f64 {
    let n = x.round();
    let r = x - n;
    let (a, b) = (r.abs(), 0.5 - r.abs());
    let s = if a <= 0.25 { a * horner::<f64, FUSED, 9>(a * a, SIN_PI) } else { horner::<f64, FUSED, 10>(b * b, COS_PI) };
    let half = n * 0.5;
    let s = s.copysign(r);
    if half != half.floor() { -s } else { s }
}

/// sin(πr)/r as a polynomial in r², highest degree first, for |r| up to 1/2, within 3e-11 relative.
const SIN_PI_PLAIN: [f64; 6] = [-0.007022080551569066, 0.08204642580715658, -0.5992513181742443, 2.550163239211499, -5.167712762079367, 3.141592653523149];

/// sin πx for the narrower types: one polynomial in r = x - n, with n the nearest whole number to x, and no branch.
#[inline(always)]
fn sinpi_plain<const FUSED: bool>(x: f64) -> f64 {
    let n = x.round_ties_even();
    let r = x - n;
    let s = r * horner::<f64, FUSED, 6>(r * r, SIN_PI_PLAIN);
    let half = n * 0.5;
    if half != half.floor() { -s } else { s }
}

/// 1 - x, or x itself from 1/2 up: the argument that the Lanczos sum takes, with Γ(x) = π / (sin πx Γ(1 - x)) below 1/2.
#[inline(always)]
fn reflected(x: f64) -> (bool, Double<f64>) { let reflect = x < 0.5; (reflect, if reflect { Double::sum(1.0, -x) } else { Double::new(x, 0.0) }) }

#[inline(always)]
fn gamma64<const FUSED: bool>(x: f64) -> f64 {
    let (reflect, y) = reflected(x);
    let (s, e) = lanczos::<FUSED>(y);
    // e^(e - g) = Γ(y)/S(y), which stays finite wherever Γ(y) is. Below 1/2, e^(g - e) is 1/Γ(y), which can be subnormal, so it carries a factor of 2^512, which the other factor gives back first, to keep its rounding from growing.
    let d = Double::sum(e.hi, -G);
    let (hi, lo) = (d.hi, d.lo + e.lo);
    let up = Double::sum(-hi, LN2_512.0);
    let p = if reflect { exp_plus::<f64, FUSED>(up.hi, up.lo - lo + LN2_512.1) } else { exp_plus::<f64, FUSED>(hi, lo) };
    // Below 1/2, π / (sin πx S(y)) is a double-word quotient, so the result rounds once more, in the last product.
    let f = Double::new(PI, PI_LO).div::<FUSED>(s.mul_float::<FUSED>(sinpi::<FUSED>(x)));
    let g = if reflect { (f.hi * TWO_M512).mla::<FUSED>(p, f.lo * TWO_M512 * p) } else { s.hi.mla::<FUSED>(p, s.lo * p) };
    if x == 0.0 { 1.0 / x } else if x < 0.0 && x == x.floor() { f64::NAN } else if x > 171.7 { f64::INFINITY } else { g }
}

#[inline(always)]
fn lgamma64<const FUSED: bool>(x: f64) -> f64 {
    let (reflect, y) = reflected(x);
    // lgamma's error counts in ulps of the larger of the result and 2, so S needs no double words here, unlike in `gamma`.
    let s = sum_plain::<FUSED>(y.hi) * EXP_NEG_G;
    let e = exponent::<FUSED>(y, false);
    // Below 1/2, ln |Γ(x)| = ln(π / |sin πx S(y) e^-g|) - e: one logarithm of one quotient. For tiny x, Γ(x) is 1/x within an ulp, and the same logarithm takes |x| instead.
    let tiny = x.abs() < 1e-17;
    let l = ln_split::<FUSED>(if tiny { x.abs() } else if reflect { PI / (sinpi::<FUSED>(x).abs() * s) } else { s });
    let r = l.plus(if reflect { -e } else { e });
    if x.is_infinite() || (x <= 0.0 && x == x.floor()) { f64::INFINITY } else if tiny { -l.hi } else { r.hi + r.lo }
}

/// S(y) for y ≥ 1/2, in plain floats, which lose about 3 ulps.
#[inline(always)]
fn sum_plain<const FUSED: bool>(y: f64) -> f64 { let v = 1.0 / y; horner::<f64, FUSED, 13>(v, NUM) / horner::<f64, FUSED, 13>(v, DEN) }

/// Γ(x) for the narrower types, in plain f64.
#[inline(always)]
fn gamma_plain<const FUSED: bool>(x: f64) -> f64 {
    let reflect = x < 0.5;
    let y = if reflect { 1.0 - x } else { x };
    let s = sum_plain::<FUSED>(y);
    let p = exp_narrow::<FUSED>((y - 0.5) * (ln::<f64, FUSED>(y + G_HALF) - 1.0) - G);
    let g = if reflect { PI / (sinpi_plain::<FUSED>(x) * s * p) } else { s * p };
    if x == 0.0 { 1.0 / x } else if x < 0.0 && x == x.floor() { f64::NAN } else { g }
}

/// y + 8 and y (y+1) ... (y+7) below 8, and y and 1 from 8 up. Γ(y) = Γ(z)/P moves y to where Stirling's series converges fast.
#[inline(always)]
fn shifted(y: f64) -> (f64, f64) {
    let p = (y * (y + 1.0)) * ((y + 2.0) * (y + 3.0)) * ((y + 4.0) * (y + 5.0)) * ((y + 6.0) * (y + 7.0));
    if y < 8.0 { (y + 8.0, p) } else { (y, 1.0) }
}

/// ln(2π)/2 plus Stirling's series after its leading terms, at w = 1/z: ln Γ(z) - (z - 1/2) ln z + z, within 3e-10 from z = 8 up.
#[inline(always)]
fn stirling_tail<const FUSED: bool>(w: f64) -> f64 { w.mla::<FUSED>(horner::<f64, FUSED, 3>(w * w, [STIRLING[7], STIRLING[8], STIRLING[9]]), HALF_LN_2PI) }

/// ln |Γ(x)| for the narrower types, in plain f64, from Stirling's series at (z, P) = `shifted(y)`. Below 1/2, ln |Γ(x)| = ln(π / |sin πx|) - ln Γ(1 - x). The shift's product and the reflection's quotient share one logarithm.
#[inline(always)]
fn lgamma_stirling<const FUSED: bool>(x: f64) -> f64 {
    let reflect = x < 0.5;
    let (z, p) = shifted(if reflect { 1.0 - x } else { x });
    let s = (z - 0.5).mla::<FUSED>(ln_narrow::<FUSED>(z), stirling_tail::<FUSED>(1.0 / z) - z);
    let (num, den) = if reflect { (PI * p, sinpi_plain::<FUSED>(x).abs()) } else { (1.0, p) };
    let l = ln_narrow::<FUSED>(num / den);
    let r = if reflect { l - s } else { s + l };
    if x.is_infinite() || (x <= 0.0 && x == x.floor()) { f64::INFINITY } else { r }
}

// erfc follows FreeBSD's s_erf.c at f64 and s_erff.c at f32, through compiler-builtins' libm. Each polynomial is highest degree first.
/// One width's coefficients. erf(x) = x + x pp(x²)/qq(x²) for |x| below 0.84375, and erf(1 + s) = erx + pa(s)/qa(s) up to 1.25. Beyond, erfc(x) = e^(-x² - 0.5625 + r(v)/s(v))/x for v = 1/x², with `ra` and `sa` up to 1/0.35 and `rb` and `sb` above. `rb` and `sb` start with a zero, so that they match `ra` and `sa` in length.
struct Erf { erx: f64, pp: [f64; 5], qq: [f64; 6], pa: [f64; 7], qa: [f64; 7], ra: [f64; 8], sa: [f64; 9], rb: [f64; 8], sb: [f64; 9] }

const ERF64: Erf = Erf { erx: 8.45062911510467529297e-01,
    pp: [-2.37630166566501626084e-05, -5.77027029648944159157e-03, -2.84817495755985104766e-02, -3.25042107247001499370e-01, 1.28379167095512558561e-01],
    qq: [-3.96022827877536812320e-06, 1.32494738004321644526e-04, 5.08130628187576562776e-03, 6.50222499887672944485e-02, 3.97917223959155352819e-01, 1.0],
    pa: [
        -2.16637559486879084300e-03, 3.54783043256182359371e-02, -1.10894694282396677476e-01, 3.18346619901161753674e-01, -3.72207876035701323847e-01,
        4.14856118683748331666e-01, -2.36211856075265944077e-03,
    ], qa: [
        1.19844998467991074170e-02, 1.36370839120290507362e-02, 1.26171219808761642112e-01, 7.18286544141962662868e-02, 5.40397917702171048937e-01,
        1.06420880400844228286e-01, 1.0,
    ], ra: [
        -9.81432934416914548592e+00, -8.12874355063065934246e+01, -1.84605092906711035994e+02, -1.62396669462573470355e+02, -6.23753324503260060396e+01,
        -1.05586262253232909814e+01, -6.93858572707181764372e-01, -9.86494403484714822705e-03,
    ], sa: [
        -6.04244152148580987438e-02, 6.57024977031928170135e+00, 1.08635005541779435134e+02, 4.29008140027567833386e+02, 6.45387271733267880336e+02,
        4.34565877475229228821e+02, 1.37657754143519042600e+02, 1.96512716674392571292e+01, 1.0,
    ], rb: [
        0.0, -4.83519191608651397019e+02, -1.02509513161107724954e+03, -6.37566443368389627722e+02, -1.60636384855821916062e+02, -1.77579549177547519889e+01,
        -7.99283237680523006574e-01, -9.86494292470009928597e-03,
    ], sb: [
        0.0, -2.24409524465858183362e+01, 4.74528541206955367215e+02, 2.55305040643316442583e+03, 3.19985821950859553908e+03, 1.53672958608443695994e+03,
        3.25792512996573918826e+02, 3.03380607434824582924e+01, 1.0,
    ] };

const ERF32: Erf = Erf { erx: 0.8450629115104675,
    pp: [-2.3763017452438362e-05, -0.005770270247012377, -0.028481749817728996, -0.32504209876060486, 0.12837916612625122],
    qq: [-3.9602282413397916e-06, 0.0001324947370449081, 0.0050813062116503716, 0.06502225250005722, 0.3979172110557556, 1.0], pa: [
        -0.002166375517845154, 0.03547830507159233, -0.11089469492435455, 0.31834661960601807, -0.3722078800201416, 0.41485610604286194, -0.0023621185682713985,
    ], qa: [0.011984500102698803, 0.01363708358258009, 0.12617121636867523, 0.07182865589857101, 0.5403979420661926, 0.10642088204622269, 1.0], ra: [
        -9.814329147338867, -81.28743743896484, -184.60508728027344, -162.39666748046875, -62.37533187866211, -10.558626174926758, -0.6938585638999939,
        -0.009864944033324718,
    ], sa: [
        -0.06042441353201866, 6.570249557495117, 108.63500213623047, 429.0081481933594, 645.3872680664062, 434.5658874511719, 137.6577606201172,
        19.65127182006836, 1.0,
    ],
    rb: [0.0, -483.5191955566406, -1025.0950927734375, -637.5664672851562, -160.63638305664062, -17.75795555114746, -0.7992832660675049, -0.009864943102002144],
    sb: [0.0, -22.44095230102539, 474.5285339355469, 2553.05029296875, 3199.858154296875, 1536.7296142578125, 325.7925109863281, 30.33806037902832, 1.0] };

#[inline(always)]
fn erf_table<R: Native>() -> &'static Erf { if R::WIDE { &ERF64 } else { &ERF32 } }

/// Horner's rule at `x` with the coefficients of `a` where `first` holds and of `b` elsewhere. Choosing each coefficient, instead of branching between two polynomials, keeps a loop that calls it free of branches.
#[inline(always)]
fn horner_either<R: Native, const FUSED: bool, const N: usize>(x: R, first: bool, a: [f64; N], b: [f64; N]) -> R {
    let pick = |i: usize| if first { R::narrow(a[i]) } else { R::narrow(b[i]) };
    let mut y = pick(0);
    for i in 1..N { y = y.mla::<FUSED>(x, pick(i)); }
    y
}

/// (erf(x) - x)/x for |x| below 0.84375.
#[inline(always)]
fn erf_small<R: Native, const FUSED: bool>(x: R) -> R {
    let t = erf_table::<R>();
    let z = x * x;
    horner::<R, FUSED, 5>(z, t.pp) / horner::<R, FUSED, 6>(z, t.qq)
}

/// erfc(a) for a from `ERFC_SPLIT` up.
#[inline(always)]
fn erfc_tail<R: Native, const FUSED: bool>(a: R) -> R {
    let t = erf_table::<R>();
    let one = R::one();
    either(
        a < R::narrow(1.25),
        #[inline(always)]
        || {
            let s = a - one;
            (one - R::narrow(t.erx)) - horner::<R, FUSED, 7>(s, t.pa) / horner::<R, FUSED, 7>(s, t.qa)
        },
        #[inline(always)]
        || {
            let v = one / (a * a);
            let near = a < R::narrow(1.0 / 0.35);
            let rs = horner_either::<R, FUSED, 8>(v, near, t.ra, t.rb) / horner_either::<R, FUSED, 9>(v, near, t.sa, t.sb);
            // z has the high half of a's bits, so z² is exact, and -a² = -z² + (z - a)(z + a) holds as a double word.
            let z = a.upper();
            let h = Double::sum(-z * z, R::narrow(-0.5625));
            let u = Double::sum(h.hi, (z - a) * (z + a) + rs);
            exp_plus::<R, FUSED>(u.hi, u.lo + h.lo) / a
        },
    )
}

/// `first()` where `use_first` holds, and `second()` elsewhere. f32, and f64 on x86-64, compute both and select, so that a loop that calls it vectorizes. Elsewhere f64 computes only the one it needs. On the Mac, computing both made f64 `erfc` 1.3 times slower, because two f64 lanes in a NEON vector don't repay computing every region. On CI's x86-64 runner, the branching f64 `erfc` cost 2.7 times as many copies as on the Mac, where AVX2 holds four lanes and a mispredicted branch costs more.
#[inline(always)]
fn either<R: Native>(use_first: bool, first: impl Fn() -> R, second: impl Fn() -> R) -> R {
    if R::WIDE && !cfg!(target_arch = "x86_64") { return if use_first { first() } else { second() }; }
    let (a, b) = (first(), second());
    if use_first { a } else { b }
}

/// Arm Optimized Routines' erff: erf(x) = x + x a(x²) below 0.875, and 1 - e^-(|x| + |x| b(|x|)) up to 4. Each polynomial is highest degree first. Neither needs a division, which makes it faster at f32 than FreeBSD's rational functions.
const ERF_A: [f64; 6] = [-0.0005991049692966044, 0.004993395879864693, -0.026766734197735786, 0.11281802505254745, -0.37612485885620117, 0.12837915122509003];
const ERF_B: [f64; 7] =
    [1.729489304125309e-05, -0.00038320868043228984, 0.003883934346958995, -0.024254560470581055, 0.1067778468132019, 0.6348466873168945, 0.12871751189231873];
/// The same two formulas for f64, below 1.25 and up to 6, from weighted least-squares fits to erf(x)/x - 1 and -ln erfc(x)/x - 1.
const ERF_A64: [f64; 13] = [
    4.649419429428393e-11, -1.042430407003087e-09, 1.435492841519301e-08, -1.629337626683206e-07, 1.6454188634626338e-06, -1.492505258803544e-05,
    0.0001205530210596214, -0.0008548325957658537, 0.0052239776019798794, -0.026866170642096086, 0.11283791670935252, -0.37612638903183276, 0.12837916709551256,
];
const ERF_B64: [f64; 14] = [
    1.1301887528580086e-10, -5.139418309129716e-09, 1.0625456517811032e-07, -1.334618914272856e-06, 1.1433194544113143e-05, -7.070387057856521e-05,
    0.0003228431634442731, -0.0010706392328036834, 0.0022821868790661145, -0.0007443657653741413, -0.01877248218879148, 0.10260955401906749, 0.63666577565146,
    0.12837314142093484,
];

/// erf(x) = x + x a(x²), for |x| below the split of the coefficients `a`.
#[inline(always)]
fn erf_near<R: Native, const FUSED: bool, const A: usize>(x: R, a: [f64; A]) -> R { horner::<R, FUSED, A>(x * x, a).mla::<FUSED>(x, x) }

/// erf by the formulas of `ERF_A` and `ERF_B`, with coefficients `a` below `split` and `b` up to `top`, beyond which it is ±1. Both formulas are computed, so that a loop that calls it has no branch.
#[inline(always)]
fn erf_poly<R: Native, const FUSED: bool, const A: usize, const B: usize>(x: R, a: [f64; A], b: [f64; B], split: f64, top: f64) -> R {
    let m = x.abs();
    let small = erf_near::<R, FUSED, A>(x, a);
    let large = (R::one() - exp::<R, FUSED>(-horner::<R, FUSED, B>(m, b).mla::<FUSED>(m, m))).copysign(x);
    if x.is_nan() { x } else if m < R::narrow(split) { small } else if m < R::narrow(top) { large } else { R::one().copysign(x) }
}

/// The error function, by `erf_poly` at each width.
#[inline(always)]
pub(crate) fn erf<R: Native, const FUSED: bool>(x: R) -> R {
    if R::WIDE { erf_poly::<R, FUSED, 13, 14>(x, ERF_A64, ERF_B64, 1.25, 6.0) } else { erf_poly::<R, FUSED, 6, 7>(x, ERF_A, ERF_B, 0.875, 4.0) }
}

/// 1 - erf(x), computed at the argument's own width.
#[inline(always)]
pub(crate) fn erfc<R: Native, const FUSED: bool>(x: R) -> R {
    erfc_ends(x, either(x.abs() < R::narrow(ERFC_SPLIT), #[inline(always)] || erfc_near::<R, FUSED>(x), #[inline(always)] || erfc_far::<R, FUSED>(x)))
}

/// The |x| below which `erfc_near` holds, and from which `erfc_far` does.
const ERFC_SPLIT: f64 = 0.84375;

/// erfc(x) for |x| below `ERFC_SPLIT`.
#[inline(always)]
fn erfc_near<R: Native, const FUSED: bool>(x: R) -> R {
    let (one, half) = (R::one(), R::narrow(0.5));
    let y = erf_small::<R, FUSED>(x);
    if x < R::narrow(0.25) { one - (x + x * y) } else { half - (x - half + x * y) }
}

/// erfc(x) for |x| from `ERFC_SPLIT` up.
#[inline(always)]
fn erfc_far<R: Native, const FUSED: bool>(x: R) -> R { let t = erfc_tail::<R, FUSED>(x.abs()); if x < R::zero() { R::narrow(2.0) - t } else { t } }

/// erfc(x) from `r`, a formula's value, except that NaN gives NaN and a magnitude from 28 up gives 0 or 2.
#[inline(always)]
fn erfc_ends<R: Native>(x: R, r: R) -> R {
    if x.is_nan() { x } else if x.abs() < R::narrow(28.0) { r } else if x < R::zero() { R::narrow(2.0) } else { R::zero() }
}

/// Giles's single-precision approximation of erfinv(p)/p, as polynomials highest degree first: `GILES` in w - 2.5 for w = -ln(1 - p²) below 5, and `GILES_TAIL` in √w - 3 up to 16.
const GILES: [f64; 9] =
    [2.81022636e-08, 3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503, -0.00417768164, 0.246640727, 1.50140941];
const GILES_TAIL: [f64; 9] =
    [-0.000200214257, 0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613, 0.00943887047, 1.00167406, 2.83297682];
/// erfinv(p)/p as a polynomial in √w - 7 for w from 16 up to the logarithm of the least f32 subnormal, where `GILES_TAIL` no longer holds. It was fitted with mpmath's `chebyfit`, and is within 1e-8 relative.
const GILES_FAR: [f64; 11] = [
    3.1297268653862105e-10, -2.430591888561262e-09, 7.967095370797346e-09, -4.234119381819269e-08, 3.1378449112095126e-07, -1.6691356665594907e-06,
    6.066115530379893e-06, 6.369503371198156e-06, -0.0005126738823338601, 1.008595581302738, 6.869019655969833,
];

/// erfinv(p) from p and d = 1 - p², by Giles's approximation and `GILES_FAR`. `either` chooses among the three polynomials.
#[inline(always)]
fn giles<R: Native, const FUSED: bool>(p: R, d: R) -> R {
    let w = -ln::<R, FUSED>(d);
    let central = || horner::<R, FUSED, 9>(w - R::narrow(2.5), GILES);
    let tail = || horner::<R, FUSED, 9>(w.sqrt() - R::narrow(3.0), GILES_TAIL);
    let far = || horner::<R, FUSED, 11>(w.sqrt() - R::narrow(7.0), GILES_FAR);
    let x = p * either(w < R::narrow(5.0), central, || either(w < R::narrow(16.0), tail, far));
    if d == R::zero() { R::infinity().copysign(p) } else { x }
}

/// The x with erf(x) = p. f32 takes Giles's approximation. f64 takes Boost.Math's, through `maddock`.
#[inline(always)]
pub(crate) fn erfinv<R: Native, const FUSED: bool>(p: R) -> R {
    let one = R::one();
    if R::WIDE { R::narrow(erfinv64::<FUSED>(p.wide())) } else { giles::<R, FUSED>(p, (one - p) * (one + p)) }
}

/// The x with erfc(x) = q, computed as `erfinv` is.
#[inline(always)]
pub(crate) fn erfcinv<R: Native, const FUSED: bool>(q: R) -> R {
    let one = R::one();
    if R::WIDE { R::narrow(erfcinv64::<FUSED>(q.wide())) } else { giles::<R, FUSED>(one - q, q * (R::narrow(2.0) - q)) }
}

#[inline(always)]
fn erfinv64<const FUSED: bool>(p: f64) -> f64 {
    let a = p.abs();
    if a < 1.0 { maddock::<FUSED>(a, 1.0 - a).copysign(p) } else if a == 1.0 { f64::INFINITY.copysign(p) } else { f64::NAN }
}

/// For q above 1, the result is -erfcinv(2 - q), by the reflection erfc(-x) = 2 - erfc(x). Both halves share one call to `maddock`, with arguments |q - 1| and min(q, 2 - q). A branch between the halves would be mispredicted on uniform q.
#[inline(always)]
fn erfcinv64<const FUSED: bool>(q: f64) -> f64 {
    if !(q > 0.0 && q < 2.0) { return if q == 0.0 { f64::INFINITY } else if q == 2.0 { f64::NEG_INFINITY } else { f64::NAN }; }
    maddock::<FUSED>((q - 1.0).abs(), q.min(2.0 - q)).copysign(1.0 - q)
}

/// [`erfc`](crate::erfc) as a function object, with the slice form `many`, which runs `erfc_blocks` on a slice.
#[derive(Clone, Copy, Debug)]
pub struct Erfc;
forms!(Erfc at real: |_, [x]| crate::slice::Total(erfc::<_, { crate::FUSED }>(x)),
many |_, [x], n, out| match x {
    Arg::Many(x) => erfc_blocks::<_, { crate::FUSED }>(&x[..n], out),
    x => each!([x], n, out, |[x]| crate::slice::Total(erfc::<_, { crate::FUSED }>(x))),
});

/// `erfc` on each item of `x`, written into `out`, a block of `BLOCK` items at a time. A block whose magnitudes all lie below `ERFC_SPLIT` computes only `erfc_near`, and one whose magnitudes all reach it computes only `erfc_far`. A mixed block runs `erfc`, which computes both on f32 and on x86-64 f64.
fn erfc_blocks<R: Native, const FUSED: bool>(x: &[R], out: &mut [MaybeUninit<R>]) -> bool {
    const BLOCK: usize = 256;
    #[inline(always)]
    fn fill<R: Native>(xs: &[R], os: &mut [MaybeUninit<R>], f: impl Fn(R) -> R) { for (o, &v) in os.iter_mut().zip(xs) { o.write(f(v)); } }
    let split = R::narrow(ERFC_SPLIT);
    for (xs, os) in x.chunks(BLOCK).zip(out.chunks_mut(BLOCK)) {
        let (low, high) = xs.iter().fold((R::infinity(), R::zero()), |(low, high), &v| (low.min(v.abs()), high.max(v.abs())));
        if high < split { fill(xs, os, #[inline(always)] |v| erfc_ends(v, erfc_near::<R, FUSED>(v))); } else if low >= split { fill(xs, os, #[inline(always)] |v| erfc_ends(v, erfc_far::<R, FUSED>(v))); } else { fill(xs, os, erfc::<R, FUSED>); }
    }
    true
}
/// [`erfinv`](crate::erfinv) as a function object, with the slice form `many`, which runs `regions` on a slice of f64s.
#[derive(Clone, Copy, Debug)]
pub struct Erfinv;
forms!(Erfinv at real: |_, [p]| erfinv::<_, { crate::FUSED }>(p),
    many |_, [p], n, out| inverses(p, n, out, #[inline(always)] |p| erfinv::<_, { crate::FUSED }>(p), #[inline(always)] |p| (p.abs(), 1.0 - p.abs(), p)));

/// [`erfcinv`](crate::erfcinv) as a function object, with the slice form `many`, which runs `regions` on a slice of f64s.
#[derive(Clone, Copy, Debug)]
pub struct Erfcinv;
forms!(Erfcinv at real: |_, [q]| erfcinv::<_, { crate::FUSED }>(q),
    many |_, [q], n, out| inverses(q, n, out, #[inline(always)] |q| erfcinv::<_, { crate::FUSED }>(q), #[inline(always)] |q| ((q - 1.0).abs(), q.min(2.0 - q), 1.0 - q)));

/// The slice form of [`Erfinv`] and [`Erfcinv`]: `regions` on a slice of f64s, and `f` on each item otherwise. `parts` gives an f64 item's `maddock` arguments and the value whose sign its result takes.
fn inverses<R: Native>(x: Arg<'_, R>, n: usize, out: &mut [MaybeUninit<R>], f: impl Fn(R) -> R, parts: impl Fn(f64) -> (f64, f64, f64)) -> bool {
    match x { Arg::Many(x) if R::WIDE => regions::<R, { crate::FUSED }>(&x[..n], out, f, parts), x => each!([x], n, out, |[x]| f(x)) }
}

/// `maddock` on each item of `x`, written into `out`, with no branch on an item's region. In each block of `BLOCK` items, one pass places the central items' positions from the front and the others' from the back, and a second pass splits the others the same way by whether q is at least 1/4. Each region's formula then runs on its items' arguments, gathered into one buffer. On uniform arguments, `maddock`'s branches were mispredicted about three times in four, which took half its time. Items outside the domain, and NaN, take `f`. Gives `false` where an output is NaN and its item isn't.
fn regions<R: Native, const FUSED: bool>(x: &[R], out: &mut [MaybeUninit<R>], f: impl Fn(R) -> R, parts: impl Fn(f64) -> (f64, f64, f64)) -> bool {
    const BLOCK: usize = 512;
    let (mut place, mut split, mut t) = ([0u16; BLOCK], [0u16; BLOCK], [0.0; BLOCK]);
    let mut ok = true;
    for (xs, os) in x.chunks(BLOCK).zip(out.chunks_mut(BLOCK)) {
        let item = |i: u16| parts(xs[usize::from(i)].wide());
        let (mut lo, mut hi) = (0, xs.len());
        for i in 0..xs.len() as u16 {
            let central = item(i).0 <= 0.5;
            (place[lo], place[hi - 1]) = (i, i);
            lo += usize::from(central);
            hi -= usize::from(!central);
        }
        let (central, others) = place[..xs.len()].split_at(lo);
        let (mut up, mut down) = (0, others.len());
        for &i in others {
            let upper = item(i).1 >= 0.25;
            (split[up], split[down - 1]) = (i, i);
            up += usize::from(upper);
            down -= usize::from(!upper);
        }
        let (upper, below) = split[..others.len()].split_at(up);
        gathered(central, &mut t, os, &item, #[inline(always)] |(p, _, _)| p, #[inline(always)] |t| {
            for v in t { *v = maddock_central::<FUSED>(*v) }
        });
        gathered(upper, &mut t, os, &item, #[inline(always)] |(_, q, _)| q, #[inline(always)] |t| {
            for v in t { *v = maddock_upper::<FUSED>(*v) }
        });
        gathered(below, &mut t, os, &item, #[inline(always)] |(_, q, _)| q, #[inline(always)] |t| {
            for v in t.iter_mut() { *v = maddock_g::<FUSED>(*v) }
            for v in t { *v = maddock_below::<FUSED>(*v) }
        });
        for (i, (&v, o)) in xs.iter().zip(os.iter_mut()).enumerate() {
            if !(item(i as u16).0 < 1.0) {
                let r = f(v);
                o.write(r);
                ok &= !r.is_nan() | v.is_nan();
            }
        }
    }
    ok
}

/// `formula` on the argument that `read` takes from each item at `positions`, gathered into `t`. Each result goes to its item's place in `os`, with the item's sign.
#[inline(always)]
fn gathered<R: Native>(
    positions: &[u16], t: &mut [f64], os: &mut [MaybeUninit<R>], item: impl Fn(u16) -> (f64, f64, f64), read: impl Fn((f64, f64, f64)) -> f64,
    formula: impl Fn(&mut [f64])) {
    let t = &mut t[..positions.len()];
    for (v, &i) in t.iter_mut().zip(positions) { *v = read(item(i)) }
    formula(t);
    for (&v, &i) in t.iter().zip(positions) { os[usize::from(i)].write(R::narrow(v.copysign(item(i).2))); }
}

/// erfinv(p) for p from 0 to 1, given p and q = 1 - p, by John Maddock's rational approximations in Boost.Math's 64-bit `erf_inv_imp`. They need no iteration. Each region gives y + r, for a constant y and a ratio r of polynomials. Up to p = 1/2, the result is g(y + r) with g = p(p + 10). Beyond p = 1/2, the regions read only q. Both callers compute it exactly there. From q = 1/4 up, the result is √(-2 ln q) / (y + r). Below q = 1/4, the result is g(y + r) with g = √(-ln q), and g also chooses the region.
#[inline(always)]
fn maddock<const FUSED: bool>(p: f64, q: f64) -> f64 {
    if p <= 0.5 { maddock_central::<FUSED>(p) } else if q >= 0.25 { maddock_upper::<FUSED>(q) } else { maddock_below::<FUSED>(maddock_g::<FUSED>(q)) }
}

/// `maddock` up to p = 1/2.
#[inline(always)]
fn maddock_central<const FUSED: bool>(p: f64) -> f64 { erfinv_regions::CENTRAL.scaled::<FUSED>(p * (p + 10.0), p) }

/// `maddock` for q from 1/4 to 1/2.
#[inline(always)]
fn maddock_upper<const FUSED: bool>(q: f64) -> f64 { use erfinv_regions::UPPER; (-2.0 * ln::<f64, FUSED>(q)).sqrt() / (UPPER.y + UPPER.ratio::<FUSED>(q)) }

/// √(-ln q), which chooses among `maddock`'s regions below q = 1/4.
#[inline(always)]
fn maddock_g<const FUSED: bool>(q: f64) -> f64 { (-ln::<f64, FUSED>(q)).sqrt() }

/// `maddock` below q = 1/4, from g = √(-ln q).
#[inline(always)]
fn maddock_below<const FUSED: bool>(g: f64) -> f64 {
    use erfinv_regions::*;
    if g < 3.0 { BELOW_3.scaled::<FUSED>(g, g) } else if g < 6.0 { BELOW_6.scaled::<FUSED>(g, g) } else if g < 18.0 { BELOW_18.scaled::<FUSED>(g, g) } else { BELOW_44.scaled::<FUSED>(g, g) }
}

/// One region of `maddock`: polynomials `p` and `q`, highest degree first, in x - `shift`, and the constant `y` that their ratio corrects.
struct Region<const P: usize, const Q: usize> { y: f64, shift: f64, p: [f64; P], q: [f64; Q] }

impl<const P: usize, const Q: usize> Region<P, Q> {
    /// P(x - shift) / Q(x - shift).
    #[inline(always)]
    fn ratio<const FUSED: bool>(&self, x: f64) -> f64 { let s = x - self.shift; horner::<f64, FUSED, P>(s, self.p) / horner::<f64, FUSED, Q>(s, self.q) }

    /// g y + g r, for the ratio r at x, as Boost.Math computes it.
    #[inline(always)]
    fn scaled<const FUSED: bool>(&self, g: f64, x: f64) -> f64 { g.mla::<FUSED>(self.ratio::<FUSED>(x), g * self.y) }
}

/// The regions of `maddock`, from Boost.Math's `erf_inv.hpp`. `CENTRAL` serves p up to 1/2, and `UPPER` q from 1/4 to 1/2. Below q = 1/4, each region serves √(-ln q) below the bound in its name. At f64, √(-ln q) is at most 27.3, at the least subnormal q. Only wider floats reach Boost.Math's region beyond 44. This module leaves it out.
mod erfinv_regions {
    use super::Region;
    pub(super) const CENTRAL: Region<8, 10> = Region { y: 0.08913147449493408, shift: 0.0, p: [
        -0.005387729650712429, 0.008226878746769157, 0.02198786811111689, -0.03656379714117627, -0.012692614766297404, 0.03348066254097446,
        -0.008368748197417368, -0.0005087819496582806,
    ], q: [
        0.0008862163904564247, -0.0023339375937419, 0.07952836873415717, -0.05273963823400997, -0.7122890234154284, 0.662328840472003, 1.5622155839842302,
        -1.5657455823417585, -0.9700050433032906, 1.0,
    ] };
    pub(super) const UPPER: Region<9, 9> = Region { y: 2.249481201171875, shift: 0.25, p: [
        -3.6719225470772936, 21.12946554483405, 17.445385985570866, -44.6382324441787, -18.851064805871424, 17.644729840837403, 8.3705032834312,
        0.10526468069939171, -0.20243350835593876,
    ], q: [
        1.7211476576120028, -22.643693341313973, 10.826866735546016, 48.560921310873994, -20.14326346804852, -28.66081804998, 3.971343795334387,
        6.242641248542475, 1.0,
    ] };
    pub(super) const BELOW_3: Region<11, 8> = Region { y: 0.807220458984375, shift: 1.125, p: [
        -6.81149956853777e-10, 2.8522533178221704e-08, -6.794655751811263e-07, 0.0021455899538880526, 0.029015791000532906, 0.14286953440815717,
        0.3377855389120359, 0.38707973897260434, 0.11703015634199525, -0.16379404719331705, -0.1311027816799519,
    ], q: [0.011059242293464892, 0.15226433829533179, 0.848854343457902, 2.5930192162362027, 4.778465929458438, 5.381683457070069, 3.4662540724256723, 1.0] };
    pub(super) const BELOW_6: Region<9, 7> = Region { y: 0.9399557113647461, shift: 3.0, p: [
        2.6633922742578204e-12, -2.304047769118826e-10, 4.60469890584318e-06, 0.00015754461742496055, 0.0018712349281955923, 0.009508047013259196,
        0.018557330651423107, -0.0022242652921344794, -0.0350353787183178,
    ], q: [7.646752923027944e-05, 0.00263861676657016, 0.03415891436709477, 0.22009110576413124, 0.7620591645536234, 1.3653349817554064, 1.0] };
    pub(super) const BELOW_18: Region<9, 7> = Region { y: 0.9836282730102539, shift: 6.0, p: [
        9.905570997331033e-17, -2.811287356288318e-14, 4.625961635228786e-09, 4.4969678992770644e-07, 1.4962478375834237e-05, 0.00020938631748758808,
        0.001056288621524929, -0.0011295143874558028, -0.016743100507663373,
    ], q: [2.82243172016108e-07, 2.7533547476472603e-05, 0.0009640118070051656, 0.016074608709367652, 0.1381518657490833, 0.5914293448864175, 1.0] };
    pub(super) const BELOW_44: Region<8, 7> = Region { y: 0.9971456527709961, shift: 18.0, p: [
        -1.1676501239718427e-18, 1.4559628671867504e-12, 4.116328311909442e-10, 3.963410113048012e-08, 1.6239777734251093e-06, 2.5472303741302746e-05,
        -7.79190719229054e-06, -0.002497821279189813,
    ], q: [5.097612765997785e-10, 1.4443775662814415e-07, 1.4500735981823264e-05, 0.0006905382656226846, 0.01694108381209759, 0.2071231122144225, 1.0] };
}

/// The most steps that a series or continued fraction takes. Each stops sooner, once a step changes its result by less than an f64 ulp at f64, or by less than `NARROW_EPS` at the narrower types.
const STEPS: usize = 100_000;
/// Stands in for 0 in the continued fractions, which would otherwise divide by it.
const TINY: f64 = 1e-300;
/// Where the narrower types stop a series or continued fraction: 2^-26, about a sixteenth of an f32 ulp.
const NARROW_EPS: f64 = 1.0 / (1u64 << 26) as f64;
/// 2^52, beyond which the continued fraction's convergents are rescaled.
const RESCALE: f64 = 4503599627370496.0;

/// x^a e^-x / Γ(a), the factor before the incomplete gamma function's series and continued fraction. Its exponent stays a double word until the exponential, so its large terms cancel without rounding.
fn gamma_factor<const FUSED: bool>(a: f64, x: f64) -> f64 {
    let t = ln_dd::<FUSED>(Double::new(x, 0.0), true).mul_float::<FUSED>(a).plus(Double::new(-x, 0.0));
    let t = t.plus(-lgamma_dd::<FUSED>(Double::new(a, 0.0)));
    exp_plus::<f64, FUSED>(t.hi, t.lo)
}

/// `gamma_factor` for the narrower types, in plain f64, for a > 0. With (z, P) = `shifted(a)`, Stirling's series turns it into exp(a ln(x/z) + ln(z (P/z^(z-a))²)/2 + z - x - `stirling_tail`(1/z)). z - a is 0 or 8, so the second logarithm's argument needs only products, and f32's precision for its logarithm.
fn gamma_factor_plain<const FUSED: bool>(a: f64, x: f64) -> f64 {
    let (z, p) = shifted(a);
    let w = 1.0 / z;
    let u = if z > a {
        let w2 = w * w;
        let w4 = w2 * w2;
        p * (w4 * w4)
    } else { 1.0 };
    let half_log = 0.5 * ln_narrow::<FUSED>(z * u * u);
    let e = a.mla::<FUSED>(ln::<f64, FUSED>(x * w), half_log + (z - x) - stirling_tail::<FUSED>(w));
    exp_narrow::<FUSED>(e)
}

/// The regularized lower and upper incomplete gamma functions P(a, x) and Q(a, x) = 1 - P(a, x), at f64 if `wide` and for the narrower types otherwise. Below x = a + 1 a series gives P, and above it a continued fraction gives Q. The other is 1 minus it, which keeps its relative accuracy, because each method covers the range where its result is the smaller one.
fn incomplete_gamma<const FUSED: bool>(a: f64, x: f64, wide: bool) -> (f64, f64) {
    if !(a > 0.0 && a.is_finite() && x >= 0.0) { return (f64::NAN, f64::NAN); }
    if x == 0.0 { return (0.0, 1.0); }
    if x.is_infinite() { return (1.0, 0.0); }
    let eps = if wide { f64::EPSILON } else { NARROW_EPS };
    let factor = || if wide { gamma_factor::<FUSED>(a, x) } else { gamma_factor_plain::<FUSED>(a, x) };
    if x < a + 1.0 {
        // Each step adds two terms, x/(n+1) and x²/((n+1)(n+2)) times the last, from one division.
        let (mut term, mut sum, mut n) = (1.0 / a, 1.0 / a, a);
        for _ in 0..STEPS {
            let r = x / ((n + 1.0) * (n + 2.0));
            let first = term * r * (n + 2.0);
            term *= r * x;
            sum += first + term;
            n += 2.0;
            if term < sum * eps { break; }
        }
        let p = sum * factor();
        return (p, 1.0 - p);
    }
    // Q / factor = 1/(x + 1 - a - 1(1 - a)/(x + 3 - a - 2(2 - a)/(x + 5 - a - ...))), by its convergents p/q, as Cephes' `igamc` computes them. A step finds p and q without dividing. Only the convergence test divides, once every two steps.
    let (mut y, mut z) = (1.0 - a, x + 2.0 - a);
    let (mut p0, mut q0, mut p1, mut q1) = (1.0, x, x + 1.0, z * x);
    let mut h = p1 / q1;
    for i in (1..STEPS).step_by(2) {
        for k in i..i + 2 {
            y += 1.0;
            z += 2.0;
            let yk = y * k as f64;
            (p0, q0, p1, q1) = (p1, q1, p1 * z - p0 * yk, q1 * z - q0 * yk);
        }
        // A power of 2 rescales the convergents exactly before they overflow.
        if p1.abs() > RESCALE { (p0, q0, p1, q1) = (p0 / RESCALE, q0 / RESCALE, p1 / RESCALE, q1 / RESCALE); }
        if q1 != 0.0 {
            let r = p1 / q1;
            let done = (h - r).abs() <= eps * r.abs();
            h = r;
            if done { break; }
        }
    }
    let q = h * factor();
    (1.0 - q, q)
}

/// P(a, x) and Q(a, x) at every type. The narrower types stop iterating at their own precision.
#[inline(always)]
pub(crate) fn gammainc<R: Native, const FUSED: bool>(a: R, x: R) -> R { R::narrow(incomplete_gamma::<FUSED>(a.wide(), x.wide(), R::WIDE).0) }

#[inline(always)]
pub(crate) fn gammaincc<R: Native, const FUSED: bool>(a: R, x: R) -> R { R::narrow(incomplete_gamma::<FUSED>(a.wide(), x.wide(), R::WIDE).1) }

fn gammainc64<const FUSED: bool>(a: f64, x: f64) -> f64 { incomplete_gamma::<FUSED>(a, x, true).0 }

/// A Halley step of a root finder for P(x) = p, with P' = `slope` and P''/P' = `curve`. Halley's method converges cubically, so a step below `SETTLED` of x leaves x within an ulp.
#[inline]
fn halley(x: f64, residual: f64, slope: f64, curve: f64) -> f64 { let u = residual / slope; x - u / (1.0 - 0.5 * (u * curve).min(1.0)) }
const SETTLED: f64 = 1.5e-8;

/// The x with P(a, x) = p: a first guess from the Wilson–Hilferty approximation for a > 1, or from P's leading terms for a ≤ 1, then Halley steps.
fn gammaincinv64<const FUSED: bool>(a: f64, p: f64) -> f64 {
    if !(a > 0.0 && a.is_finite() && (0.0..=1.0).contains(&p)) { return f64::NAN; }
    if p == 0.0 { return 0.0; }
    if p == 1.0 { return f64::INFINITY; }
    let lg = lgamma64::<FUSED>(a);
    let a1 = a - 1.0;
    let mut x = if a > 1.0 {
        let t = (-2.0 * ln::<f64, FUSED>(p.min(1.0 - p))).sqrt();
        let z = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        let z = if p < 0.5 { -z } else { z };
        let c = 1.0 - 1.0 / (9.0 * a) - z / (3.0 * a.sqrt());
        (a * c * c * c).max(1e-3)
    } else {
        let t = 1.0 - a * (0.253 + a * 0.12);
        if p < t { exp::<f64, FUSED>(ln::<f64, FUSED>(p / t) / a) } else { 1.0 - ln::<f64, FUSED>(1.0 - (p - t) / (1.0 - t)) }
    };
    for _ in 0..12 {
        if x <= 0.0 { return 0.0; }
        let lx = ln::<f64, FUSED>(x);
        let density = exp::<f64, FUSED>(a1 * lx - x - lg);
        let next = halley(x, gammainc64::<FUSED>(a, x) - p, density, a1 / x - 1.0);
        let step = x - next;
        x = if next <= 0.0 { 0.5 * x } else { next };
        if step.abs() < SETTLED * x { break; }
    }
    x
}

/// The continued fraction of the incomplete beta function, by Lentz's method, which converges fast for x below (a + 1)/(a + b + 2).
fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    let clamp = |v: f64| if v.abs() < TINY { TINY } else { v };
    let (mut c, mut d) = (1.0, 1.0 / clamp(1.0 - (a + b) * x / (a + 1.0)));
    let mut h = d;
    for m in 1..STEPS {
        let (m, m2) = (m as f64, 2.0 * m as f64);
        let even = m * (b - m) * x / ((a - 1.0 + m2) * (a + m2));
        d = 1.0 / clamp(1.0 + even * d);
        c = clamp(1.0 + even / c);
        h *= d * c;
        let odd = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2));
        d = 1.0 / clamp(1.0 + odd * d);
        c = clamp(1.0 + odd / c);
        let step = d * c;
        h *= step;
        if (step - 1.0).abs() < f64::EPSILON { break; }
    }
    h
}

/// ln B(a, b) = ln Γ(a) + ln Γ(b) - ln Γ(a + b).
fn lbeta<const FUSED: bool>(a: f64, b: f64) -> f64 { lgamma64::<FUSED>(a) + lgamma64::<FUSED>(b) - lgamma64::<FUSED>(a + b) }

/// The regularized incomplete beta function I_x(a, b). Above (a + 1)/(a + b + 2) it computes 1 - I_(1-x)(b, a), where the continued fraction converges fast.
fn betainc64<const FUSED: bool>(a: f64, b: f64, x: f64) -> f64 {
    if !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite() && (0.0..=1.0).contains(&x)) { return f64::NAN; }
    if x == 0.0 || x == 1.0 { return x; }
    // a ln x + b ln(1 - x) - ln B(a, b), with 1 - x and a + b exact as double words, and every term a double word until the exponential.
    let (one, sum) = (Double::new(1.0, 0.0), Double::sum(a, b));
    let (lx, l1x) = (ln_dd::<FUSED>(Double::new(x, 0.0), true), ln_dd::<FUSED>(one.plus(Double::new(-x, 0.0)), true));
    let t = lx.mul_float::<FUSED>(a).plus(l1x.mul_float::<FUSED>(b));
    let t = t.plus(lgamma_dd::<FUSED>(sum)).plus(-lgamma_dd::<FUSED>(Double::new(a, 0.0)).plus(lgamma_dd::<FUSED>(Double::new(b, 0.0))));
    let front = exp_plus::<f64, FUSED>(t.hi, t.lo);
    if x < (a + 1.0) / (a + b + 2.0) { front * beta_fraction(a, b, x) / a } else { 1.0 - front * beta_fraction(b, a, 1.0 - x) / b }
}

/// The x with I_x(a, b) = p. A guess above 1/2 solves I_(1-x)(b, a) = 1 - p instead, because x there has too few digits for the steps to settle.
fn betaincinv64<const FUSED: bool>(a: f64, b: f64, p: f64) -> f64 {
    if !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite() && (0.0..=1.0).contains(&p)) { return f64::NAN; }
    if p == 0.0 || p == 1.0 { return p; }
    let x = beta_guess::<FUSED>(a, b, p);
    if x > 0.5 { 1.0 - beta_steps::<FUSED>(b, a, 1.0 - p, 1.0 - x) } else { beta_steps::<FUSED>(a, b, p, x) }
}

/// A first guess at the x with I_x(a, b) = p: from a normal approximation for a and b from 1 up, or from I's leading terms near 0 and 1 otherwise.
fn beta_guess<const FUSED: bool>(a: f64, b: f64, p: f64) -> f64 {
    if a >= 1.0 && b >= 1.0 {
        let t = (-2.0 * ln::<f64, FUSED>(p.min(1.0 - p))).sqrt();
        let z = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        let z = if p < 0.5 { -z } else { z };
        let al = (z * z - 3.0) / 6.0;
        let h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let w = z * (al + h).sqrt() / h - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
        a / (a + b * exp::<f64, FUSED>(2.0 * w))
    } else {
        let (la, lb) = (ln::<f64, FUSED>(a / (a + b)), ln::<f64, FUSED>(b / (a + b)));
        let (t, u) = (exp::<f64, FUSED>(a * la) / a, exp::<f64, FUSED>(b * lb) / b);
        let w = t + u;
        if p < t / w { exp::<f64, FUSED>(ln::<f64, FUSED>(a * w * p) / a) } else { 1.0 - exp::<f64, FUSED>(ln::<f64, FUSED>(b * w * (1.0 - p)) / b) }
    }
}

/// Halley steps from `x` toward the x with I_x(a, b) = p.
fn beta_steps<const FUSED: bool>(a: f64, b: f64, p: f64, mut x: f64) -> f64 {
    let (a1, b1, lb) = (a - 1.0, b - 1.0, lbeta::<FUSED>(a, b));
    for _ in 0..20 {
        if x <= 0.0 || x >= 1.0 { return x.clamp(0.0, 1.0); }
        let density = exp::<f64, FUSED>(a1 * ln::<f64, FUSED>(x) + b1 * log1p::<f64, FUSED>(-x) - lb);
        let next = halley(x, betainc64::<FUSED>(a, b, x) - p, density, a1 / x - b1 / (1.0 - x));
        let step = x - next;
        x = if next <= 0.0 { 0.5 * x } else if next >= 1.0 { 0.5 * (x + 1.0) } else { next };
        if step.abs() < SETTLED * x { break; }
    }
    x
}
