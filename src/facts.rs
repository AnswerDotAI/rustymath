//! The facts that frontends need about rustymath's function objects, in one table: the reduction by which each function folds numbers, the reduction by which it folds Booleans, and the rule by which it takes characters. [`facts!`](crate::facts) gives the table to a frontend's macro, which turns the rows into what that frontend needs.

/// Calls `$then!` with rustymath's table of facts, after the further arguments in parentheses. The table has six sections:
/// - `folds`: each reduction in [`reduce`](crate::reduce) by which functions fold numbers, with the BPL primitives and the Python names of those functions;
/// - `booleans`: the same for folds of Booleans, where a Python name is a function of Booleans only;
/// - `comparisons`: each comparison in [`tolerant`](crate::tolerant), with the BPL primitive it serves. A comparison takes characters by [`Rule::Ordered`](crate::characters::Rule::Ordered), and each frontend makes it with the tolerance it needs;
/// - `characters`: each other object that takes characters, with its [`Rule`](crate::characters::Rule), the arguments that make it, and the BPL primitives it serves;
/// - `products`: each kernel in [`linalg::Product`](crate::linalg::Product) that a product `f.g` names, with the pairs of BPL primitives and the pairs of Python names that name it;
/// - `boolean_products`: the same for products of Booleans, which a frontend looks up first where both arguments are Booleans. BPL's `+.∧` counts, while NumPy's Boolean `matmul` gives OR-AND.
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
    } };
}
