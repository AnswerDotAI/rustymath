use super::*;

pub(super) fn map_real<K, F, Dst, const N: usize>(
    object: &K, variant: crate::domain::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, ops: &NativeOps<{ FUSED }>, repair: bool,
    judge: crate::domain::Variant, output: &mut Buffer) -> Result<(), Error> where K: crate::facts::Formula<N>,
    F: crate::Native + crate::buffer::Cast + Source<Dst> + crate::tolerant::Banded, Dst: Element + crate::Float {
    if !K::BOOLEAN && repair && object.repair_routine().is_some() {
        return map_blocks::<F, Dst, N>(inputs, range, variant.checked, output, &|args, n, out| {
            let ok = crate::facts::real_slice(object, ops, args, n, out);
            if judge.checked && K::FAILS_NAN && !ok { Err(Kind::Domain) } else { Ok(()) }
        });
    }
    if const { K::BOOLEAN } {
        let bands = object.bands().and_then(|(t, left, right)| Some((crate::facts::run_tolerance::<F, _>(ops, t)?, left, right)));
        return map_written::<F, bool, N>(inputs, range, variant.checked, judge.checked, output, |args, n, out| {
            if let Some((t, left, right)) = bands && let [x, y] = args.as_slice()
                && crate::tolerant::against(t, *x, *y, n, out, left, right, F::banded).is_some() { return; }
            crate::slice::mapped(args, n, out, #[inline(always)] |x| (object.real(ops, x).unwrap_or(F::zero()) != F::zero(), true));
        }, |x, r| verdict(object, judge, ops, x, if r { F::one() } else { F::zero() }));
    }
    map_values::<F, Dst, N>(
        inputs, range, variant.checked, judge.checked, output, #[inline(always)] |x| Source::<Dst>::read(object.real(ops, x).unwrap_or(F::zero())),
        |x, r| verdict(object, judge, ops, x, F::narrow(r.wide())))
}
