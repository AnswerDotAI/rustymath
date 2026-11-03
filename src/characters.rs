//! Character arithmetic uses ordinary numerical kernels on borrowed Unicode code points. Numbers order before characters.
use crate::buffer::{DType, Ints, View, Width};
use crate::error::{Error, Kind};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule { Identity, Ordered, Shift, Difference }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByKinds { Truth(bool), Left, Right }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy { Codes { computing: DType, result: DType }, Kinds(ByKinds) }

pub(crate) fn policy(
    rule: Option<Rule>, types: [DType; 2], numeric_result: DType, kinds: impl FnOnce([i32; 2]) -> Option<i32>) -> Result<Option<Policy>, Error> {
    let [xc, yc] = types.map(|t| t == DType::Character);
    if !xc && !yc { return Ok(None); }
    let (computing, result) = match (rule, xc, yc) {
        (Some(Rule::Ordered), true, true) => (DType::Integer(Width::I32), if numeric_result == DType::Boolean { DType::Boolean } else { DType::Character }),
        (Some(Rule::Ordered), _, _) => {
            let result = kinds([i32::from(xc), i32::from(yc)]).ok_or_else(|| Error::new(Kind::Domain, "ordered character function has no integer body"))?;
            return Ok(Some(Policy::Kinds(if numeric_result == DType::Boolean { ByKinds::Truth(result != 0) }
            else if (result == 1) == xc { ByKinds::Left }
            else { ByKinds::Right })));
        }
        (Some(Rule::Difference), true, true) => (DType::Integer(Width::I32), DType::Integer(Width::I32)),
        (Some(Rule::Shift), true, false) | (Some(Rule::Shift), false, true) | (Some(Rule::Difference), true, false)
            if !types.iter().any(|t| matches!(t, DType::Complex(_))) =>
        { (DType::Integer(Width::I64), DType::Character) }
        _ => return Err(Error::new(Kind::Domain, "function does not take these character arguments")),
    };
    Ok(Some(Policy::Codes { computing, result }))
}

pub fn order_of_kinds(x_character: bool, y_character: bool) -> Ordering { x_character.cmp(&y_character) }

/// Borrows each character's code point without copying.
pub fn codes(chars: &[char]) -> &[i32] {
    // SAFETY: char has u32's size and alignment, with every value below 0x110000.
    unsafe { std::slice::from_raw_parts(chars.as_ptr().cast(), chars.len()) }
}

pub fn as_codes(view: View<'_>) -> View<'_> { match view { View::Characters(c) => View::Integers(Ints::I32(codes(c))), view => view } }
