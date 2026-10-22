//! The facts that frontends need about rustymath's function objects, in one table: the reduction by which each function folds numbers, the reduction by which it folds Booleans, and the rule by which it takes characters. [`facts!`](crate::facts) gives the table to a frontend's macro, which turns the rows into what that frontend needs.

/// Calls `$then!` with rustymath's table of facts, after the further arguments in parentheses. The table has four sections:
/// - `folds`: each reduction in [`reduce`](crate::reduce) by which functions fold numbers, with the BPL primitives and the Python names of those functions;
/// - `booleans`: the same for folds of Booleans, where a Python name is a function of Booleans only;
/// - `comparisons`: each comparison in [`tolerant`](crate::tolerant), with the BPL primitive it serves. A comparison takes characters by [`Rule::Ordered`](crate::characters::Rule::Ordered), and each frontend makes it with the tolerance it needs;
/// - `characters`: each other object that takes characters, with its [`Rule`](crate::characters::Rule), the arguments that make it, and the BPL primitives it serves.
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
    } };
}
