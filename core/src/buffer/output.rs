use super::*;
use crate::{Arg, parallel::Slots};

pub trait OutputItem: Element { fn slots<'s, 'b>(output: &'s mut MapOutput<'_, 'b>) -> Option<&'s mut Slots<'b, Self>>; }

macro_rules! outputs {
    ([$($v:ident: ($dtype:expr) $t:ty),+]) => {
        /// Where a map appends: an owned buffer or the next slots of a caller's result.
        pub enum MapOutput<'a, 'b> { Buffer(&'a mut Buffer), $($v(&'a mut Slots<'b, $t>)),+ }
        impl<'b> MapOutput<'_, 'b> {
            fn dtype(&self) -> DType { match self { Self::Buffer(b) => b.dtype(), $(Self::$v(_) => <$t>::DTYPE),+ } }
            pub fn reborrow(&mut self) -> MapOutput<'_, 'b> { match self { Self::Buffer(b) => MapOutput::Buffer(b), $(Self::$v(s) => MapOutput::$v(s)),+ } }
            pub fn fill(&mut self, n: usize) {
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
element_types!((outputs));

/// Conversion buffers retained for one execution, by computing dtype and operand position.
#[derive(Default)]
pub struct Scratch { buffers: Vec<(DType, Vec<Buffer>)>, results: Vec<Buffer>, copied: usize }

impl Scratch {
    fn buffers(&mut self, dtype: DType, n: usize) -> &mut [Buffer] {
        let at = self.buffers.iter().position(|(stored, _)| *stored == dtype).unwrap_or_else(|| {
            self.buffers.push((dtype, Vec::new()));
            self.buffers.len() - 1
        });
        let buffers = &mut self.buffers[at].1;
        buffers.resize_with(n.max(buffers.len()), || Buffer::with_capacity(dtype, 0));
        &mut buffers[..n]
    }
    pub fn copied(&self) -> usize { self.copied }
    pub fn record_copies(&mut self, items: usize) { self.copied += items; }
    fn result<T: Element>(&mut self) -> &mut Vec<T> {
        let at = self.results.iter().position(|buffer| buffer.dtype() == T::DTYPE).unwrap_or_else(|| {
            self.results.push(Buffer::with_capacity(T::DTYPE, 0));
            self.results.len() - 1
        });
        T::vec_mut(&mut self.results[at]).expect("a result buffer has its computing dtype")
    }
    pub fn buffer(&mut self, dtype: DType) -> &mut Buffer { &mut self.buffers(dtype, 1)[0] }
    pub fn inputs<T: Element, const N: usize>(&mut self) -> [&mut Vec<T>; N] {
        let buffers: &mut [Buffer; N] = self.buffers(T::DTYPE, N).try_into().expect("scratch has an entry per operand");
        buffers.each_mut().map(|buffer| T::vec_mut(buffer).expect("scratch keeps its computing dtype"))
    }
}

pub struct Output<'a, 'b> { pub target: MapOutput<'a, 'b>, pub result_dtype: DType, pub checked: bool, pub scratch: Option<&'a mut Scratch> }

impl Output<'_, '_> {
    /// The block body is erased before output conversion.
    pub fn write<O: OutputItem>(
        &mut self, n: usize, apply: &(dyn Fn(std::ops::Range<usize>, &mut Slots<'_, O>, Option<&mut Scratch>) -> bool + Sync)) -> Result<(), Error> {
        let checked = self.checked;
        if let MapOutput::Buffer(buffer) = &mut self.target {
            if let Some(values) = O::vec_mut(buffer) { extended(values, n, self.scratch.as_deref_mut(), apply); } else {
                with_dtype!(buffer.dtype(), T => {
                    let values = T::vec_mut(buffer).expect("a buffer has its dtype");
                    extended(values, n, self.scratch.as_deref_mut(), &|part, slots, scratch| {
                        converted::<O>(&mut slots.into(), part, checked, scratch, apply).map_err(crate::error::record).is_ok()
                    });
                });
            }
            return Ok(());
        }
        if let Some(slots) = O::slots(&mut self.target) { apply(0..n, slots, self.scratch.as_deref_mut()); return Ok(()); }
        converted(&mut self.target, 0..n, checked, self.scratch.as_deref_mut(), apply)
    }
}

fn extended<O: Element>(
    values: &mut Vec<O>, n: usize, scratch: Option<&mut Scratch>,
    apply: &(dyn Fn(std::ops::Range<usize>, &mut Slots<'_, O>, Option<&mut Scratch>) -> bool + Sync)) {
    match scratch {
        Some(scratch) => { Slots::append(values, n, |slots| apply(0..n, slots, Some(scratch))); }
        None => { crate::parallel::extended(values, n, crate::CHUNK, &|part, slots| apply(part, slots, None)); }
    }
}

fn converted<O: OutputItem>(
    target: &mut MapOutput<'_, '_>, range: std::ops::Range<usize>, checked: bool, mut work: Option<&mut Scratch>,
    apply: &(dyn Fn(std::ops::Range<usize>, &mut Slots<'_, O>, Option<&mut Scratch>) -> bool + Sync)) -> Result<(), Error> {
    let mut values = work.as_deref_mut().map_or_else(Vec::new, |work| std::mem::take(work.result::<O>()));
    let conversion = crate::convert::Conversion { input: O::DTYPE, output: target.dtype(), kind: crate::convert::ConversionKind::Truncate };
    let result = (|| {
        for at in (range.start..range.end).step_by(crate::CHUNK) {
            let n = crate::CHUNK.min(range.end - at);
            values.clear();
            Slots::append(&mut values, n, |slots| apply(at..at + n, slots, work.as_deref_mut()));
            if let Err(error) = conversion.append(O::view(&values), checked, crate::tolerant::Tolerance::EXACT, work.as_deref_mut(), target.reborrow()) {
                target.fill(range.end - at - n);
                return Err(error);
            }
        }
        Ok(())
    })();
    if let Some(work) = work { *work.result::<O>() = values; }
    result
}
