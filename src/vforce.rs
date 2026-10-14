//! vForce, Accelerate's array functions, which implement `SinCos` on macOS.
use crate::complex::SinCos;
use num_complex::Complex;
use std::mem::MaybeUninit;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vvsincos(sin: *mut f64, cos: *mut f64, y: *const f64, n: *const i32);
    fn vvsincosf(sin: *mut f32, cos: *mut f32, y: *const f32, n: *const i32);
    fn vvcosisin(z: *mut Complex<f64>, y: *const f64, n: *const i32);
    fn vvcosisinf(z: *mut Complex<f32>, y: *const f32, n: *const i32);
}

/// `n` as the item count that vForce takes.
fn count(n: usize) -> i32 { i32::try_from(n).expect("vForce takes at most `i32::MAX` items") }

macro_rules! sin_cos {
    ($($t:ty: $sincos:ident, $cosisin:ident;)+) => {$(
        impl SinCos for $t {
            fn sin_cos_many(ys: &[$t], sin: &mut Vec<$t>, cos: &mut Vec<$t>) {
                let n = ys.len();
                sin.reserve(n);
                cos.reserve(n);
                // SAFETY: each vector has room for `n` more items, which vForce writes before the lengths grow to cover them.
                unsafe {
                    $sincos(sin.spare_capacity_mut().as_mut_ptr().cast(), cos.spare_capacity_mut().as_mut_ptr().cast(), ys.as_ptr(), &count(n));
                    sin.set_len(sin.len() + n);
                    cos.set_len(cos.len() + n);
                }
            }

            fn cis(ys: &[$t], out: &mut [MaybeUninit<Complex<$t>>], _: &mut (Vec<$t>, Vec<$t>)) {
                let out = &mut out[..ys.len()];
                // SAFETY: `out` holds a slot for each item of `ys`, and `Complex` has the layout of C's complex type.
                unsafe { $cosisin(out.as_mut_ptr().cast(), ys.as_ptr(), &count(ys.len())) }
            }
        }
    )+};
}
sin_cos! { f32: vvsincosf, vvcosisinf; f64: vvsincos, vvcosisin; }
