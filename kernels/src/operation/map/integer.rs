use super::*;

pub(super) fn map_integer<K, I, const N: usize>(
    object: &K, variant: crate::domain::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, judge: crate::domain::Variant,
    output: &mut Buffer) -> Result<(), Error> where K: crate::facts::Formula<N>, I: crate::recipe::Overflowing {
    let ops = crate::recipe::IntegerOps::<I>::new(variant);
    let value = |x| object.integer(&ops, x).unwrap_or(I::FILL);
    if const { K::BOOLEAN } {
        return map_values::<I, bool, N>(
            inputs, range, variant.checked, judge.checked, output, |x| Into::<i64>::into(value(x)) != 0,
            |x, r| verdict(object, judge, &ops, x, I::low(i64::from(r))));
    }
    map_values::<I, I, N>(inputs, range, variant.checked, judge.checked, output, value, |x, r| verdict(object, judge, &ops, x, r))
}
