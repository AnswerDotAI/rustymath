use super::*;
use crate::buffer::Cast;

#[derive(Clone, Copy)]
pub(crate) struct Inputs<'a, 'b>(pub(crate) &'a [crate::agreement::Read<'b>]);

impl<'a, 'b> Inputs<'a, 'b> {
    pub(crate) fn len(self) -> usize { self.0.len() }
    pub(crate) fn selected(self, k: usize) -> Self { Self(&self.0[k..k + 1]) }
    pub(crate) fn real(self) -> bool { self.0.iter().all(|r| r.sources.iter().all(|s| !matches!(s.view.dtype(), DType::Complex(_)))) }
    pub(crate) fn in_place<F: Cast>(self, range: &std::ops::Range<usize>) -> bool { self.0.iter().all(|r| r.in_place::<F>(range)) }
    pub(crate) fn borrowed<F: Cast, const N: usize>(self, range: std::ops::Range<usize>, checked: bool) -> Result<Option<[Arg<'b, F>; N]>, Error> {
        let mut args = [Arg::Many(&[]); N];
        for (arg, read) in args.iter_mut().zip(self.0) {
            let Some(items) = read.borrowed(range.clone(), checked)? else { return Ok(None) };
            *arg = items;
        }
        Ok(Some(args))
    }
    pub(crate) fn part<'s, F: Cast, const N: usize>(
        self, range: std::ops::Range<usize>, checked: bool, scratch: &'s mut [&mut Vec<F>; N]
    ) -> Result<([Arg<'s, F>; N], usize), Error> where 'a: 's, 'b: 's {
        let (mut args, mut copied) = ([Arg::Many(&[]); N], 0);
        for (k, (arg, scratch)) in args.iter_mut().zip(scratch).enumerate() {
            let (items, n) = self.0[k].part_counted(range.clone(), checked, scratch)?;
            *arg = items;
            copied += n;
        }
        Ok((args, copied))
    }
}
