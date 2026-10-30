//! The facts that frontends need about rustymath's function objects. [`facts!`](crate::facts) gives one table to a frontend's macro, which turns its rows into what that frontend needs: the reduction by which each function folds numbers, the reduction by which it folds Booleans, and the rule by which it takes characters. [`Facts`] gives what a planner can know about an object's results before it reads any item: where no result is negative, and where no result fails. basedpl's kernel table passes each object that it runs to its planner (`pervasive::Survey`), which reads these facts, so that a call whose result type depends on its arguments' values, as the square root's does under dynamic kinds, can stay in a trace or a fused stage. A fact never changes a result.
//!
//! A non-negative value here is never negative and never ¯0, since `1 ÷ ¯0` is ¯∞. It can be NaN: a NaN argument gives a NaN result, which no function counts as a failure.
use crate::reduce::{Maximum, Minimum};
use crate::Flipped;

/// The facts about a function object of `N` arguments. The defaults say that nothing is known.
pub trait Facts<const N: usize> {
    /// Whether every result is non-negative, where `nonnegative` marks the arguments that are, and `constants` gives each argument that is one number.
    fn nonnegative(&self, _nonnegative: [bool; N], _constants: [Option<f64>; N]) -> bool { false }
    /// Whether no result fails where the arguments that `nonnegative` marks are non-negative. `constants` gives each argument that is one number. The square root and the logarithm never fail on non-negative arguments. A root of a constant odd degree never fails.
    fn total_on(&self, _nonnegative: [bool; N], _constants: [Option<f64>; N]) -> bool { false }
    /// The argument that every result equals, 0 for the left and 1 for the right, where `constants` gives each argument that is one number: `x×1` keeps its left argument, and `1×x` its right. A constant of ¯0 counts as 0, so `x+0` keeps `x` where `x` is ¯0, whose sum with +0 is +0.
    fn keeps(&self, _constants: [Option<f64>; N]) -> Option<usize> { None }
}

/// The argument that `constants` keep, where `identity` gives, for each side, the constant there that keeps the other argument, as 0 does on either side of `+`.
#[inline]
fn kept(constants: [Option<f64>; 2], identity: [Option<f64>; 2]) -> Option<usize> {
    match constants { [_, Some(c)] if Some(c) == identity[1] => Some(0), [Some(c), _] if Some(c) == identity[0] => Some(1), _ => None }
}

/// `Facts` with the defaults, for each object of `$n` arguments, with the constants that keep the other argument (`kept`) where they're given.
macro_rules! unknown {
    ($n:literal: $($object:ty $(= $identity:expr)?),+) => {$( impl Facts<$n> for $object { $(#[inline] fn keeps(&self, c: [Option<f64>; $n]) -> Option<usize> { kept(c, $identity) })? } )+};
}
/// `Facts` for each object of `$n` arguments whose every result is non-negative.
macro_rules! always {
    ($n:literal: $($object:ty),+) => {$( impl Facts<$n> for $object { fn nonnegative(&self, _: [bool; $n], _: [Option<f64>; $n]) -> bool { true } } )+};
}
/// `Facts` for each object of `$n` arguments whose results are non-negative where every argument is, with the constants that keep the other argument (`kept`) where they're given.
macro_rules! follows {
    ($n:literal: $($object:ty $(= $identity:expr)?),+) => {$( impl Facts<$n> for $object {
        fn nonnegative(&self, x: [bool; $n], _: [Option<f64>; $n]) -> bool { x.iter().all(|&a| a) }
        $(#[inline] fn keeps(&self, c: [Option<f64>; $n]) -> Option<usize> { kept(c, $identity) })?
    } )+};
}
/// `Facts` for each object of `$n` arguments that never fails where every argument is non-negative, as a logarithm doesn't.
macro_rules! total {
    ($n:literal: $($object:ty),+) => {$( impl Facts<$n> for $object { fn total_on(&self, x: [bool; $n], _: [Option<f64>; $n]) -> bool { x.iter().all(|&a| a) } } )+};
}

always!(1: crate::Abs, crate::Exp, crate::Cosh, crate::Acos, crate::Acosh, crate::Sqrt1mx2, crate::Sqrt1px2, crate::Sqrtx2m1);
always!(2: crate::Hypot);
follows!(1: crate::Reciprocal, crate::Conj, crate::PiTimes, crate::Factorial, crate::Cbrt, crate::Sinh, crate::Tanh, crate::Asinh, crate::Atan, crate::Asin, crate::Atanh, crate::Expm1);
follows!(2: crate::Add = [Some(0.0), Some(0.0)], crate::Multiply = [Some(1.0), Some(1.0)], crate::Divide = [None, Some(1.0)], crate::MathishDivide = [None, Some(1.0)],
    crate::PiTimes, Maximum, Minimum, crate::Beta);
unknown!(1: crate::Negate, crate::Sin, crate::Cos, crate::Tan);
unknown!(2: crate::Subtract = [None, Some(0.0)], crate::WholeDivide = [None, Some(1.0)], crate::WrappingPow = [None, Some(1.0)], crate::Binom, crate::tolerant::Residue,
    crate::tolerant::FloorDivide, crate::tolerant::MathishFloorDivide, crate::tolerant::Gcd, crate::tolerant::Lcm, crate::Bitwise, crate::Nand, crate::Nor, crate::BitAnd,
    crate::BitOr, crate::BitXor, crate::LeftShift, crate::RightShift, crate::reduce::Xor, crate::reduce::Xnor, crate::tolerant::Equal, crate::tolerant::NotEqual,
    crate::tolerant::Less, crate::tolerant::LessEqual, crate::tolerant::Greater, crate::tolerant::GreaterEqual, crate::BinomDk, crate::BinomDn, crate::Atan2,
    crate::Gammainc, crate::Gammaincc, crate::Gammaincinv, crate::Polygamma);
total!(1: crate::Ln, crate::Log1p);
total!(2: crate::Logb, crate::MathishLogb);

impl Facts<1> for crate::Sqrt {
    fn nonnegative(&self, [x]: [bool; 1], _: [Option<f64>; 1]) -> bool { x }
    fn total_on(&self, [x]: [bool; 1], _: [Option<f64>; 1]) -> bool { x }
}

/// x^y, which is non-negative, and never fails, where the base is. x^1 is x.
impl Facts<2> for crate::Pow {
    fn nonnegative(&self, [x, _]: [bool; 2], _: [Option<f64>; 2]) -> bool { x }
    fn total_on(&self, [x, _]: [bool; 2], _: [Option<f64>; 2]) -> bool { x }
    #[inline] fn keeps(&self, constants: [Option<f64>; 2]) -> Option<usize> { kept(constants, [None, Some(1.0)]) }
}

/// x^k for a whole k. It is non-negative where the base is non-negative or `parity` counts the constant k as even, as it counts ∞. It keeps an argument where [`crate::Pow`] does.
impl Facts<2> for crate::WholePow {
    fn nonnegative(&self, [x, _]: [bool; 2], [_, k]: [Option<f64>; 2]) -> bool { x || k.is_some_and(|k| matches!(crate::pow::parity(k), (true, false))) }
    #[inline] fn keeps(&self, constants: [Option<f64>; 2]) -> Option<usize> { crate::Pow.keeps(constants) }
}

/// The degree-n root of y. A non-negative y gives a non-negative root and never fails. A constant odd degree, as `parity` decides it for the kernel, never fails on any y.
impl Facts<2> for crate::Root {
    fn nonnegative(&self, [_, y]: [bool; 2], _: [Option<f64>; 2]) -> bool { y }
    fn total_on(&self, [_, y]: [bool; 2], [n, _]: [Option<f64>; 2]) -> bool { y || n.is_some_and(|n| crate::pow::parity(n).1) }
}

/// `K` with its arguments in the other order, whose facts are `K`'s in the other order.
impl<K: Facts<2>> Facts<2> for Flipped<K> {
    fn nonnegative(&self, [x, y]: [bool; 2], [c, d]: [Option<f64>; 2]) -> bool { self.0.nonnegative([y, x], [d, c]) }
    fn total_on(&self, [x, y]: [bool; 2], [c, d]: [Option<f64>; 2]) -> bool { self.0.total_on([y, x], [d, c]) }
    fn keeps(&self, [c, d]: [Option<f64>; 2]) -> Option<usize> { self.0.keeps([d, c]).map(|side| 1 - side) }
}

/// `K` with its results wrapped to their integer width. It keeps an argument where `K` does, but a wrapped result can be negative where every argument is non-negative.
impl<K: Facts<2>> Facts<2> for crate::Wrapped<K> {
    fn keeps(&self, constants: [Option<f64>; 2]) -> Option<usize> { self.0.keeps(constants) }
}

/// A function's slope as a function object of its argument and result ([`Slope`](crate::derivative::Slope)), about which nothing is known.
impl<K> Facts<2> for crate::derivative::Slope<K> {}

/// Calls `$then!` with rustymath's table of facts, after the further arguments in parentheses. The table has seven sections:
/// - `folds`: each reduction in [`reduce`](crate::reduce) by which functions fold numbers, with the BPL primitives and the Python names of those functions;
/// - `booleans`: the same for folds of Booleans, where a Python name is a function of Booleans only;
/// - `comparisons`: each comparison in [`tolerant`](crate::tolerant), with the BPL primitive it serves. A comparison takes characters by [`Rule::Ordered`](crate::characters::Rule::Ordered), and each frontend makes it with the tolerance it needs;
/// - `characters`: each other object that takes characters, with its [`Rule`](crate::characters::Rule), the arguments that make it, and the BPL primitives it serves;
/// - `products`: each kernel in [`linalg::Product`](crate::linalg::Product) that a product `f.g` names, with the pairs of BPL primitives and the pairs of Python names that name it;
/// - `boolean_products`: the same for products of Booleans, which a frontend looks up first where both arguments are Booleans. BPL's `+.∧` counts, while NumPy's Boolean `matmul` gives OR-AND.
/// - `identities`: each object whose constant arguments can keep the other argument ([`Facts::keeps`]), with the BPL primitives it serves. Every object that a primitive's kernel tables choose keeps what the row's object keeps.
///
/// BPL primitives appear as patterns over basedpl's primitives, and a product's pair of them as a tuple pattern. Only basedpl's macro reads them.
#[macro_export]
macro_rules! facts {
    ($then:ident $(, $arg:tt)*) => { $then! { ($($arg),*)
        folds {
            Sum: [Arithmetic(Plus)] ["add"];
            Product: [Arithmetic(Times)] ["multiply"];
            Max: [Math(Ceiling)] ["maximum"];
            Min: [Math(Floor)] ["minimum"];
            And: [Bitwise(8)] ["bitwise_and"];
            Or: [Bitwise(14)] ["bitwise_or"];
            Xor: [Bitwise(6)] ["bitwise_xor"];
            Xnor: [Bitwise(9)] [];
        }
        booleans {
            Sum: [Arithmetic(Plus)] [];
            Max: [Math(Ceiling), Math(Gcd), Bitwise(14)] ["logical_or"];
            Min: [Math(Floor), Math(Lcm), Bitwise(8)] ["logical_and"];
            Xor: [Compare(NotEqual), Bitwise(6)] ["logical_xor"];
            Xnor: [Compare(Equal), Bitwise(9)] ["logical_xnor"];
        }
        comparisons {
            $crate::tolerant::Equal => [Compare(Equal)];
            $crate::tolerant::NotEqual => [Compare(NotEqual)];
            $crate::tolerant::Less => [Compare(Less)];
            $crate::tolerant::LessEqual => [Compare(LessEqual)];
            $crate::tolerant::Greater => [Compare(Greater)];
            $crate::tolerant::GreaterEqual => [Compare(GreaterEqual)];
        }
        characters {
            Ordered: $crate::reduce::Maximum [] => [Math(Ceiling)];
            Ordered: $crate::reduce::Minimum [] => [Math(Floor)];
            Shift: $crate::Add [] => [Arithmetic(Plus)];
            Difference: $crate::Subtract [] => [Arithmetic(Minus)];
        }
        products {
            PlusTimes: [(Arithmetic(Plus), Arithmetic(Times))] [("add", "multiply")];
            MaxPlus: [(Math(Ceiling), Arithmetic(Plus))] [("maximum", "add")];
            MinPlus: [(Math(Floor), Arithmetic(Plus))] [("minimum", "add")];
            PlusEqual: [(Arithmetic(Plus), Compare(Equal))] [("add", "equal")];
            PlusNotEqual: [(Arithmetic(Plus), Compare(NotEqual))] [("add", "not_equal")];
        }
        boolean_products {
            PlusTimes: [(Arithmetic(Plus), Math(Lcm))] [];
            OrAnd: [(Math(Gcd), Math(Lcm))] [("add", "multiply"), ("logical_or", "logical_and")];
            AndOr: [(Math(Lcm), Math(Gcd))] [("logical_and", "logical_or")];
        }
        identities {
            $crate::Add => [Arithmetic(Plus)];
            $crate::Subtract => [Arithmetic(Minus)];
            $crate::Multiply => [Arithmetic(Times)];
            $crate::Divide => [Arithmetic(Divide)];
            $crate::Pow => [Math(Power)];
        }
    } };
}
