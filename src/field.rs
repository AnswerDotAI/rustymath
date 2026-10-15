//! Arithmetic on a type whose operators take references, as `num_traits::NumAssignRef` gives them, such as `BigRational`. Each function returns a new value, and clones only the operand it starts from.
use num_traits::NumAssignRef;

/// `-x`, for a `T` whose operations include no negation.
pub(crate) fn negated<T: NumAssignRef>(x: T) -> T {
    let mut z = T::zero();
    z -= x;
    z
}

/// `x × y`.
pub(crate) fn times<T: Clone + NumAssignRef>(x: &T, y: &T) -> T {
    let mut p = x.clone();
    p *= y;
    p
}

/// `x - y`.
pub(crate) fn minus<T: Clone + NumAssignRef>(x: &T, y: &T) -> T {
    let mut d = x.clone();
    d -= y;
    d
}

/// The sum of `terms`.
pub(crate) fn sum<T: NumAssignRef>(terms: impl Iterator<Item = T>) -> T { terms.fold(T::zero(), |mut s, t| { s += t; s }) }
