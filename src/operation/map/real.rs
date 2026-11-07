use super::*;

pub(super) fn map_real<K, F, Dst, const N: usize>(
    object: &K, variant: crate::domain::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, ops: &NativeOps<{ FUSED }>, repair: bool,
    judge: crate::domain::Variant, output: &mut Buffer) -> Result<(), Error> where K: crate::facts::Formula<N>,
    F: crate::float::Native + crate::buffer::Cast + Source<Dst> + crate::tolerant::Banded, Dst: Element + crate::Float {
    if !K::BOOLEAN && repair && object.repair_routine().is_some() {
        return map_blocks::<F, Dst, N>(inputs, range, variant.checked, output, &|args, n, out| {
            let ok = real_slice(object, ops, args, n, out);
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

/// A real formula's slice body and rare-item repairs, without recording errors from intermediate values.
pub(crate) fn real_slice<K: crate::facts::Formula<N>, F: crate::float::Native + Source<O>, O: crate::Float, const N: usize>(
    object: &K, ops: &NativeOps<{ FUSED }>, args: [Arg<'_, F>; N], n: usize, out: &mut [std::mem::MaybeUninit<O>]) -> bool {
    let body = |x| object.real(ops, x).expect("a resolved object has a real body");
    if let Some(repair) = object.repair_routine() {
        let needs = |x| object.emit_repair(ops, x).unwrap_or(false);
        crate::slice::two_pass(args, n, out, |x| (!needs(x)).then(|| body(x)), |x| if needs(x) { repair.apply(x[0]) } else { body(x) })
    } else { crate::slice::each(args, n, out, #[inline(always)] |x| Source::<O>::read(body(x))) }
}
