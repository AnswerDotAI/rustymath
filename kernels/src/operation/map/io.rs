use super::*;
use crate::{buffer::Cast, parallel::Slots};

#[derive(Clone, Copy)]
pub(crate) enum Inputs<'a, 'b> { Mapped(&'a [crate::agreement::Read<'b>]), Ready(&'a [View<'b>]) }

impl<'a, 'b> Inputs<'a, 'b> {
    pub(crate) fn len(self) -> usize { match self { Self::Mapped(x) => x.len(), Self::Ready(x) => x.len() } }
    pub(crate) fn selected(self, k: usize) -> Self { match self { Self::Mapped(x) => Self::Mapped(&x[k..k + 1]), Self::Ready(x) => Self::Ready(&x[k..k + 1]) } }
    pub(crate) fn real(self) -> bool {
        let real = |v: View<'_>| !matches!(v.dtype(), DType::Complex(_));
        match self { Self::Mapped(x) => x.iter().all(|r| r.sources.iter().all(|s| real(s.view))), Self::Ready(x) => x.iter().all(|&v| real(v)) }
    }
    pub(crate) fn in_place<F: Cast>(self, range: &std::ops::Range<usize>) -> bool {
        match self { Self::Mapped(x) => x.iter().all(|r| r.in_place::<F>(range)), Self::Ready(x) => x.iter().all(|&v| F::of(v).is_some()) }
    }
    pub(crate) fn part<'s, F: Cast, const N: usize>(
        self, range: std::ops::Range<usize>, checked: bool, scratch: &'s mut [Vec<F>; N]) -> Result<[Arg<'s, F>; N], Error> where 'a: 's, 'b: 's {
        let mut args = [Arg::Many(&[]); N];
        for (k, (arg, scratch)) in args.iter_mut().zip(scratch).enumerate() {
            *arg = match self {
                Self::Mapped(x) => x[k].part(range.clone(), checked, scratch)?,
                Self::Ready(x) => {
                    let view = if x[k].len() == 1 { x[k] } else { x[k].slice(range.clone()) };
                    match F::of(view) {
                        Some(items) => Arg::of(items),
                        None => {
                            scratch.clear();
                            F::extend_cast(view, view.len(), checked, scratch)?;
                            Arg::of(scratch)
                        }
                    }
                }
            };
        }
        Ok(args)
    }
}

/// Where a map appends its results: an owned buffer or the next slots of a caller's result.
pub enum MapOutput<'a, 'b> {
    Buffer(&'a mut Buffer), F16(&'a mut Slots<'b, half::f16>), F32(&'a mut Slots<'b, f32>), F64(&'a mut Slots<'b, f64>), Bool(&'a mut Slots<'b, bool>),
    U8(&'a mut Slots<'b, u8>), I16(&'a mut Slots<'b, i16>), I32(&'a mut Slots<'b, i32>), I64(&'a mut Slots<'b, i64>),
    C32(&'a mut Slots<'b, num_complex::Complex32>), C64(&'a mut Slots<'b, num_complex::Complex64>), Char(&'a mut Slots<'b, char>)
}

pub(crate) trait OutputItem: Element { fn slots<'s, 'b>(output: &'s mut MapOutput<'_, 'b>) -> Option<&'s mut Slots<'b, Self>>; }

macro_rules! outputs {
    ($($v:ident: $t:ty),+ $(,)?) => {
        impl MapOutput<'_, '_> {
            fn dtype(&self) -> DType { match self { Self::Buffer(b) => b.dtype(), $(Self::$v(_) => <$t>::DTYPE),+ } }
            fn copy(&mut self, view: View<'_>) {
                match self { Self::Buffer(b) => { b.extend_from(view, 0..view.len()); }, $(Self::$v(s) => s.copy(<$t>::of(view).expect("result conversion has the destination's dtype"))),+ }
            }
            pub(crate) fn fill(&mut self, n: usize) {
                match self {
                    Self::Buffer(b) => with_dtype!(b.dtype(), T => { let v = T::vec_mut(b).expect("a buffer has its dtype"); v.resize(v.len() + n, T::FILL); }),
                    $(Self::$v(s) => { s.pairs(Arg::One(<$t>::FILL), Arg::One(()), n, |x, ()| (x, true)); }),+
                }
            }
        }
        $(impl OutputItem for $t {
            fn slots<'s, 'b>(output: &'s mut MapOutput<'_, 'b>) -> Option<&'s mut Slots<'b, Self>> {
                match output { MapOutput::$v(s) => Some(s), _ => None }
            }
        }
        impl<'a, 'b> From<&'a mut Slots<'b, $t>> for MapOutput<'a, 'b> {
            fn from(slots: &'a mut Slots<'b, $t>) -> Self { MapOutput::$v(slots) }
        })+
    };
}
outputs!(F16: half::f16, F32: f32, F64: f64, Bool: bool, U8: u8, I16: i16, I32: i32, I64: i64,
    C32: num_complex::Complex32, C64: num_complex::Complex64, Char: char);

pub(crate) struct Output<'a, 'b> { pub target: MapOutput<'a, 'b>, pub dtype: DType, pub checked: bool }

impl Output<'_, '_> {
    /// The block body is already erased here: output conversion never instantiates a formula at a second computing type.
    pub(crate) fn write<O: OutputItem>(&mut self, n: usize, apply: &(dyn Fn(std::ops::Range<usize>, &mut Slots<'_, O>) -> bool + Sync)) -> Result<(), Error> {
        let checked = self.checked;
        if let MapOutput::Buffer(buffer) = &mut self.target {
            if let Some(values) = O::vec_mut(buffer) { crate::parallel::extended(values, n, crate::CHUNK, apply); } else {
                with_dtype!(buffer.dtype(), T => {
                    let values = T::vec_mut(buffer).expect("a buffer has its dtype");
                    crate::parallel::extended(values, n, crate::CHUNK, &|part, slots| {
                        converted::<O>(&mut slots.into(), part, checked, apply).map_err(crate::error::record).is_ok()
                    });
                });
            }
            return Ok(());
        }
        if let Some(slots) = O::slots(&mut self.target) { apply(0..n, slots); return Ok(()); }
        converted(&mut self.target, 0..n, checked, apply)
    }
}

fn converted<O: OutputItem>(
    target: &mut MapOutput<'_, '_>, range: std::ops::Range<usize>, checked: bool, apply: &(dyn Fn(std::ops::Range<usize>, &mut Slots<'_, O>) -> bool + Sync)
) -> Result<(), Error> {
    let mut scratch = Vec::<O>::new();
    for at in (range.start..range.end).step_by(crate::CHUNK) {
        let n = crate::CHUNK.min(range.end - at);
        scratch.clear();
        Slots::append(&mut scratch, n, |slots| apply(at..at + n, slots));
        match Buffer::cast_results(O::view(&scratch), target.dtype(), checked) {
            Ok(buffer) => target.copy(buffer.view()), Err(error) => { target.fill(range.end - at); return Err(error); }
        }
    }
    Ok(())
}
