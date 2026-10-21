//! The facts that frontends need about rustymath's function objects, in one table: the reduction by which each function folds numbers, the reduction by which it folds Booleans, and the rule by which it takes characters. [`facts!`](crate::facts) gives the table to a frontend's macro, which turns the rows into what that frontend needs.

/// Calls `$then!` with rustymath's table of facts, after the further arguments in parentheses. The table has three sections:
/// - `folds`: each reduction in [`reduce`](crate::reduce) by which functions fold numbers, with the BPL primitives and the Python names of those functions;
/// - `booleans`: the same for folds of Booleans, where a Python name is a function of Booleans only;
/// - `characters`: each object that takes characters, with its [`Rule`](crate::characters::Rule), the arguments that make it, and the BPL primitives it serves.
///
/// BPL primitives appear as patterns over basedpl's primitives, which only basedpl's macro reads.
#[macro_export]
macro_rules! facts {
    ($then:ident $(, $arg:tt)*) => { $then! { ($($arg),*)
        folds {
            Sum: [Arithmetic(Plus)] ["add"];
            Product: [Arithmetic(Times)] ["multiply"];
            Max: [Math(Ceiling)] ["maximum"];
            Min: [Math(Floor)] ["minimum"];
        }
        booleans {
            Sum: [Arithmetic(Plus)] [];
            Max: [Math(Ceiling), Math(Gcd)] ["logical_or"];
            Min: [Math(Floor), Math(Lcm)] ["logical_and"];
            Xor: [Compare(NotEqual)] ["logical_xor"];
            Xnor: [Compare(Equal)] ["logical_xnor"];
        }
        characters {
            Ordered: $crate::tolerant::Equal [($crate::tolerant::Tolerance::EXACT)] => [Compare(Equal)];
            Ordered: $crate::tolerant::NotEqual [($crate::tolerant::Tolerance::EXACT)] => [Compare(NotEqual)];
            Ordered: $crate::tolerant::Less [($crate::tolerant::Tolerance::EXACT)] => [Compare(Less)];
            Ordered: $crate::tolerant::LessEqual [($crate::tolerant::Tolerance::EXACT)] => [Compare(LessEqual)];
            Ordered: $crate::tolerant::Greater [($crate::tolerant::Tolerance::EXACT)] => [Compare(Greater)];
            Ordered: $crate::tolerant::GreaterEqual [($crate::tolerant::Tolerance::EXACT)] => [Compare(GreaterEqual)];
            Ordered: $crate::reduce::Maximum [] => [Math(Ceiling)];
            Ordered: $crate::reduce::Minimum [] => [Math(Floor)];
            Shift: $crate::Add [] => [Arithmetic(Plus)];
            Difference: $crate::Subtract [] => [Arithmetic(Minus)];
        }
    } };
}
