//! Characters as their code points. A function takes characters by the rule that its object declares ([`Characters`]), and runs the object's integer kernel on the characters' code points, which [`codes`] lends without copying. Numbers order before characters ([`order_of_kinds`]), so an ordered function of numbers and characters follows from their kinds, without reading their items.
use crate::agreement::{Agreement, Mapping};
use crate::array::ArrayView;
use crate::buffer::{Buffer, DType, Element, Ints, Items, View, Width};
use crate::operation::binary_at;
use crate::slice::{Binary, Results};
use std::cmp::Ordering;

/// How a function takes characters, as its object declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// It orders by code point. A comparison gives Booleans, and the maximum or minimum gives the character with the higher or lower code point. A number orders before a character.
    Ordered,
    /// It adds an integer to a character, on either side, and gives a character.
    Shift,
    /// It subtracts. A character minus an integer gives a character, and a character minus a character gives an integer.
    Difference,
}

/// The rule by which a function object takes characters. An object without one takes no characters. Each object gets its rule from rustymath's table of facts ([`facts!`](crate::facts)).
pub trait Characters { const RULE: Rule; }
/// [`Wrapped`](crate::Wrapped) takes characters by its object's rule. Character arithmetic checks each code point after its integer kernel runs, so a wrapped integer result still fails there.
impl<K: Characters> Characters for crate::Wrapped<K> { const RULE: Rule = K::RULE; }

/// [`binary`] for objects of type `K`, as [`characters!`](crate::characters!) gives it.
pub type Kernel<K> = fn(&K, ArrayView<'_>, ArrayView<'_>, &[usize], &Agreement) -> Option<(Buffer, bool)>;

/// The [`binary`] by which `$k`'s type takes characters where it has a [`Rule`], or `None` where it takes none. As with [`kernels!`](crate::kernels), the test resolves where the macro expands, so the type must be concrete there.
#[macro_export]
macro_rules! characters {
    (of $k:expr) => {{
        #[allow(unused_imports)]
        use $crate::operation::probe::{Found as _, Missing as _};
        let found: Option<($crate::characters::Kernel<_>, bool)> =
            (&$crate::operation::probe::Probe::<_, (), $crate::operation::probe::Characters>::of(&$k)).kernel();
        found.map(|(f, _)| f)
    }};
}

macro_rules! rules {
    (() folds $folds:tt booleans $booleans:tt comparisons { $($comparison:path => [$($cbpl:tt)*];)* } characters { $($rule:ident: $object:path [$($new:tt)*] => [$($bpl:tt)*];)* } products $products:tt boolean_products $boolean_products:tt identities $identities:tt) => {
        $(impl Characters for $comparison { const RULE: Rule = Rule::Ordered; })*
        $(impl Characters for $object { const RULE: Rule = Rule::$rule; })*
    };
}
crate::facts!(rules);

/// How an item of one kind orders against an item of another, where each flag says whether its item is a character: numbers come first.
pub fn order_of_kinds(x_character: bool, y_character: bool) -> Ordering { x_character.cmp(&y_character) }

/// The code points of `chars`, lent without copying.
pub fn codes(chars: &[char]) -> &[i32] {
    // SAFETY: a `char` has the size and alignment of a `u32` and holds a value below 0x110000, so each one reads as a nonnegative `i32`.
    unsafe { std::slice::from_raw_parts(chars.as_ptr().cast(), chars.len()) }
}

/// `view` with characters read as their code points. Any other view is unchanged.
pub fn as_codes(view: View<'_>) -> View<'_> { match view { View::Characters(c) => View::Integers(Ints::I32(codes(c))), view => view } }

/// `k` on `x` and `y`, paired as `agreement` says over the result's `frame`, by `k`'s [`Rule`], with whether every result succeeded. `None` where neither argument is characters, or where the rule doesn't take arguments of these kinds. A character result that isn't a Unicode scalar value fails, as does an offset that isn't a whole number. A missing entry of a keyed character argument reads as a space, as character storage fills.
pub fn binary<K, L>(k: &K, x: ArrayView<'_>, y: ArrayView<'_>, frame: &[usize], agreement: &Agreement<L>) -> Option<(Buffer, bool)> where
    K: Characters + Binary<i32, Out: Element> + Binary<i64, Out: Element> + Sync {
    let (xc, yc) = (x.dtype() == DType::Character, y.dtype() == DType::Character);
    if !xc && !yc { return None; }
    let (xi, left) = laid(x, &agreement.left, frame, agreement.len);
    let (yi, right) = laid(y, &agreement.right, frame, agreement.len);
    let paired = Agreement { layout: (), len: agreement.len, left, right };
    let (xv, yv) = (xi.view(), yi.view());
    Some(match (K::RULE, xc, yc) {
        (Rule::Ordered, true, true) => { let (out, ok) = binary_at::<K, i32, _>(k, as_codes(xv), as_codes(yv), &paired); (characters_of(out), ok) }
        (Rule::Ordered, _, _) => kinds(k, [(xv, &paired.left), (yv, &paired.right)], xc, agreement.len),
        (Rule::Difference, true, true) => binary_at::<K, i32, _>(k, as_codes(xv), as_codes(yv), &paired),
        (Rule::Shift, true, false) | (Rule::Shift, false, true) | (Rule::Difference, true, false) => {
            let ((xs, x_ok), (ys, y_ok)) = (wide(xv)?, wide(yv)?);
            let (out, ok) = binary_at::<K, i64, _>(k, View::Integers(Ints::I64(&xs)), View::Integers(Ints::I64(&ys)), &paired);
            let View::Integers(Ints::I64(sums)) = out.view() else { unreachable!("an offset gives an i64 at i64") };
            let chars = sums.iter().map(|&s| u32::try_from(s).ok().and_then(char::from_u32)).collect::<Vec<_>>();
            let valid = chars.iter().all(Option::is_some);
            (Buffer::Characters(chars.into_iter().map(|c| c.unwrap_or(' ')).collect()), ok && x_ok && y_ok && valid)
        }
        _ => return None,
    })
}

/// An argument's items and how they lay out over the call, as `outer` and the argument's own mapping compose. A character argument whose layout reads missing entries is gathered as characters first, so that each one reads as a space.
fn laid<'a>(a: ArrayView<'a>, outer: &Mapping, frame: &[usize], len: usize) -> (Items<'a>, Mapping) {
    let (items, mapping) = a.mapped(outer, frame, a.dtype());
    let mapping = mapping.unwrap_or_else(|| outer.clone());
    if a.dtype() != DType::Character || !mapping.fills() { return (items, mapping); }
    let gathered = Buffer::cast_mapped(items.view(), &mapping, len, DType::Character).expect("characters gather as characters");
    (Items::Owned(gathered), Mapping::Linear(1))
}

/// An offset's argument at `i64`, with whether every item is whole: characters as their code points, and numbers as the whole numbers they hold. `None` for complex numbers.
fn wide(v: View<'_>) -> Option<(Vec<i64>, bool)> {
    Some(match v {
        View::Characters(c) => (codes(c).iter().map(|&c| i64::from(c)).collect(), true),
        View::Booleans(_) | View::Integers(_) => (i64::vec(Buffer::cast(v, DType::Integer(Width::I64)).ok()?)?, true),
        View::Floats(_) => match crate::structure::signed_positions(v) {
            Ok(positions) => { let whole = positions.iter().all(Option::is_some); (positions.into_iter().map(|p| p.unwrap_or(0)).collect(), whole) }
            Err(_) => (vec![0; v.len()], false),
        },
        View::Complexes(_) => return None,
    })
}

/// An ordered function's results at code points as characters: the maximum's or minimum's, which is one of its arguments' code points. A comparison's Booleans are unchanged.
fn characters_of(out: Buffer) -> Buffer {
    let View::Integers(Ints::I32(picked)) = out.view() else { return out };
    Buffer::Characters(picked.iter().map(|&c| char::from_u32(c as u32).expect("an ordered function gives one of its arguments' code points")).collect())
}

/// What an ordered function gives for a number against a character, which their kinds decide, as [`order_of_kinds`] orders them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByKinds {
    /// A comparison's result.
    Truth(bool),
    /// The maximum or minimum picks the first argument.
    Left,
    /// The maximum or minimum picks the second argument.
    Right,
}

/// [`ByKinds`] for `k`, an ordered function, where `x_character` says whether the first argument is the character. `k` runs once, on the kinds' ranks: 0 for the number and 1 for the character.
pub fn by_kinds<K: Binary<i32, Out: Element>>(k: &K, x_character: bool) -> ByKinds {
    let rank = |character: bool| i32::from(character);
    let out = <<K as Results<i32, 2>>::Out as Element>::buffer(vec![<K as Binary<i32>>::one(k, rank(x_character), rank(!x_character))]);
    match out.view() {
        View::Booleans(&[b]) => ByKinds::Truth(b),
        View::Integers(Ints::I32(&[winner])) => { if (winner == 1) == x_character { ByKinds::Left } else { ByKinds::Right } }
        _ => unreachable!("an ordered function gives a Boolean or one of its arguments"),
    }
}

/// An ordered function of numbers and characters ([`by_kinds`]): Booleans, all the same, or the argument that the maximum or minimum picks, laid over the call.
fn kinds<K: Binary<i32, Out: Element>>(k: &K, sides: [(View<'_>, &Mapping); 2], x_character: bool, len: usize) -> (Buffer, bool) {
    let (view, mapping) = match by_kinds(k, x_character) {
        ByKinds::Truth(b) => return (Buffer::Booleans(vec![b; len]), true), ByKinds::Left => sides[0], ByKinds::Right => sides[1]
    };
    (Buffer::cast_mapped(view, mapping, len, view.dtype()).expect("an argument gathers at its own dtype"), true)
}
