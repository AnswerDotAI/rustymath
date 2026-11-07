use super::*;

pub(super) fn map_complex<K, F, const N: usize>(
    object: &K, variant: crate::domain::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, ops: &NativeOps<{ FUSED }>,
    slice: Option<crate::facts::Slice<K, num_complex::Complex<F>, N, ()>>, judge: crate::domain::Variant, output: &mut Buffer) -> Result<(), Error> where
    K: crate::facts::Formula<N>, F: crate::Native + crate::buffer::Cast, num_complex::Complex<F>: crate::buffer::Cast {
    let value = |x| object.complex(ops, x).unwrap_or(num_complex::Complex::new(F::zero(), F::zero()));
    let invalid = |x, r| (!complex_valid::<K, F, N>(ops, x, r)).then_some(Kind::Domain);
    if const { K::BOOLEAN } {
        map_values::<num_complex::Complex<F>, bool, N>(
            inputs, range, variant.checked, judge.checked, output, |x| value(x).re != F::zero(),
            |x, r| invalid(x, num_complex::Complex::new(if r { F::one() } else { F::zero() }, F::zero())))
    } else if output.dtype().is_real() {
        map_values::<num_complex::Complex<F>, F, N>(
            inputs, range, variant.checked, judge.checked, output, |x| value(x).re, |x, r| invalid(x, num_complex::Complex::new(r, F::zero())))
    } else {
        map_written::<num_complex::Complex<F>, num_complex::Complex<F>, N>(inputs, range, variant.checked, judge.checked, output, |args, n, out| {
            if let Some(slice) = slice { slice(object, ops, *args, n, out); }
            else { crate::slice::mapped(args, n, out, #[inline(always)] |x| (value(x), true)); }
        }, invalid)
    }
}

/// A formula's complex results from real arguments (`Formula::real_complex`). An item that it declines takes `Formula::complex` on its arguments as complex numbers.
pub(super) fn map_real_complex<K, F, const N: usize>(
    object: &K, variant: crate::domain::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, ops: &NativeOps<{ FUSED }>,
    output: &mut Buffer) -> Result<(), Error> where K: crate::facts::Formula<N>, F: crate::Native + crate::buffer::Cast, num_complex::Complex<F>: Element {
    map_blocks::<F, num_complex::Complex<F>, N>(inputs, range, variant.checked, output, &|args, n, out| {
        let nan = num_complex::Complex::new(F::nan(), F::nan());
        if !crate::slice::mapped(&args, n, out, #[inline(always)] |x| object.real_complex(ops, x).map_or((nan, false), |z| (z, true))) {
            for (i, z) in unsafe { crate::slice::written(&mut out[..n]) }.iter_mut().enumerate() {
                let x = std::array::from_fn(|j| args[j].nth(i));
                if object.real_complex(ops, x).is_none() { *z = object.complex(ops, x.map(|v| num_complex::Complex::new(v, F::zero()))).unwrap_or(nan); }
            }
        }
        Ok(())
    })
}
