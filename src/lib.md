## Using rustymath from Rust

Add rustymath to your `Cargo.toml`:

```toml
[dependencies]
rustymath = "0.1"
```

On x86-64, build for a recent CPU, with `-C target-cpu=x86-64-v3` or `-C target-cpu=native`. You can set it once in `.cargo/config.toml`:

```toml
[target.'cfg(target_arch = "x86_64")']
rustflags = ["-C", "target-cpu=x86-64-v3"]
```

Rust's default x86-64 target has no fused multiply-add and no vector instruction for `floor` or `trunc`. Without the setting, rustymath multiplies and adds separately. Some of its loops also run one item at a time.

Each function comes in two forms. The free function, such as `exp`, is generic over `f32`, `f64` and `half::f16`. The compiler inlines it into the loop that calls it and vectorizes that loop:

```rust
use half::f16;
use rustymath::exp;

let e64 = exp(1.0f64); // 2.718281828459045
let e32 = exp(1.0f32); // 2.718282
let e16 = exp(f16::from_f32(1.0)); // 2.719
```

The function object, such as `Exp` or `Pow`, has a scalar form `one` and a slice form `many`. Both run code that rustymath compiles itself. Your build settings don't change their speed. Each argument of `many` is one item, which every result pairs with, or a slice with one item for each result. `many` appends the results to a vector and returns whether every item succeeded. A NaN result from arguments that aren't NaN counts as a failure. An object implements `Unary`, `Binary` or `Ternary` at each item type it takes: `f16`, `f32` and `f64`, complex numbers where the function has complex values, and integers for some, such as `Gcd` at `i64`. Code that is generic over the float type states the bound it needs, such as `where Exp: Unary<F, Out = F>`. The free functions need no bound.

```rust
use rustymath::{Arg, Binary, Pow};

let mut squares = Vec::new();
Pow.many(Arg::Many(&[1.5, 2.0, 3.0]), Arg::One(2.0), 3, &mut squares); // squares: [2.25, 4.0, 9.0]
```

Use the slice forms of `Sin`, `Cos` and `Tan` for arrays. They reduce each item by multiples of π with a few multiply-adds, then give the rare items beyond 2^23 at f64, or 2^20 at f32, an exact reduction in integer arithmetic. The free `sin` takes a branch for those items. That branch stops a loop that calls it from vectorizing.

`atan2(y, x)` and its object `Atan2` give the angle of the point (x, y), from -π to π. `y` comes first, as in `f64::atan2`.

The complex slice forms of `Exp`, `Ln`, `Pow`, `Sin`, `Cos`, `Sinh` and `Cosh` vectorize too. Each one splits a chunk of complex items into their real and imaginary parts and computes the results from the real slice forms. On macOS, the sines and cosines they need come from vForce, which computes both in one call. It then redoes the rare items with the scalar complex function: items with a tiny or non-finite part, and `Pow`'s special cases. `Abs`, `Angle` and `Sqrt` keep their loop over items, which runs faster than a split for their short formulas. Other complex slice forms, such as `Lgamma`'s, compute one item at a time.

`divide(x, y)` divides complex numbers, inline in the caller's loop, as the free functions do. Two real arguments divide as reals, so a real zero divisor gives ±∞ or NaN with an imaginary part of 0. Its formula's |y|² overflows beyond about 1e154 at f64 and 1e19 at f32.

### Tolerant comparison

The `tolerant` module has APL's tolerant comparison: x equals y when |x - y| is less than the tolerance times the larger of |x| and |y|. Its objects, such as `Equal`, `Less`, `Floor`, `Residue` and `Gcd`, each hold a `Tolerance`. With `Tolerance::EXACT`, a slice form runs a loop of plain IEEE comparisons, chosen once per call.

```rust
use rustymath::{tolerant::{Equal, Tolerance}, Arg, Binary};

let mut same = Vec::new();
Equal(Tolerance::DEFAULT).many(Arg::Many(&[0.3, 0.1 + 0.2, 0.31]), Arg::One(0.3), 3, &mut same); // same: [true, true, false]
```

### Other modules

- `number_theory`: primality, factorisation, next and previous primes, Euler's totient, and the enumeration and counting of primes, on a `u64`, a `u128` or a `BigUint`. Each computes in the narrowest of those that holds the value. A function that can run for a long time also has a `_with` version, whose extra argument is a check that can stop it. `factorial` gives n! as a `BigUint`.
- `optimize`: `minimize` finds the minimum of a function of an `f32` or `f64` slice, by Brent's method for one variable, Nelder–Mead, Powell's conjugate directions, L-BFGS with a gradient, or differential evolution within bounds. `solve` finds a zero of a function that gives a slice, or its least-squares solution, by Brent's method for one equation in one variable and Levenberg–Marquardt otherwise. `solve_linear` solves a linear system, or its least squares, through the matrix's products with vectors, by conjugate gradients, MINRES or restarted GMRES, with an optional preconditioner. `solve_affine` solves f(x) = 0 for an affine function the same way, through its values alone. `Method::choose` picks a method where the caller doesn't. Functions are `Objective`s and `Mapping`s: closures that return a `Result`, whose error stops the search, or types that also give a gradient, vector–Jacobian products or batches of points.
- `polynomial`: evaluation over a slice by Horner's rule, roots from the eigenvalues of the companion matrix, sorted by real part, then imaginary part, building from roots, derivatives and least-squares fitting. `value` and `factored` evaluate at one point. They, `from_roots` and `derivative` take any type with the field operations, such as exact rationals. So does `decode`, which gives the values of digits in a mixed radix.
- `reduce`: sums, products, maximums and minimums over a slice, and their running versions. `Reduction` gives each one's unit, its step for one more item, its fold of a slice, its merge of the folds of two neighbouring runs, and its scan, through `Sum`, `Product`, `Max` and `Min`. Its `checked` is the step, and whether the result fits. Float sums and products vectorize by combining the items in any order. An integer sum or product, and each running one, wraps where it doesn't fit, and reports whether it fits. `extremes` gives the least and the greatest item in one pass. `Alternating` gives the right fold by `-` through `Sum`, and by `÷` through `Product`: the sum of the pairs' differences, or the product of their ratios. A quotient of floats reports that it doesn't fit where a running product leaves the normal range. Booleans fold by `Max` and `Min` as `or` and `and`, by `Xor` and `Xnor` as not-equal and equal, and by `SumAs` as a count of trues. `Logical` scans Booleans by any function of two Booleans, given as its truth table. `SumAs<C>` also sums integers, wrapping in the type `C`, which the caller picks to hold every sum: a byte, `i16`, `i32` or `i64`, through the traits `Term` and `Tally`. `along` and `scan_along` fold and scan along the middle axis of an array of shape `[outer, len, inner]` held row by row: down the columns of a matrix, for example, with `[1, rows, columns]`. `fold_into` folds the same array one block at a time, for a caller that computes the array in blocks. A row of fewer than 16 items, such as an image's 3 channels, has a loop of its own for each length. A running maximum or minimum of integers repeats its value over each block of 256 items that leaves it unchanged.
- `linalg`: inner products of row-major matrices through the `Inner` trait: the matrix product `PlusTimes`, which also takes complex matrices, the max-plus and min-plus products `MaxPlus` and `MinPlus`, the counts of equal and unequal items `PlusEqual` and `PlusNotEqual`, and the Boolean products `OrAnd` and `AndOr`. Each numeric pair also takes `i64` matrices, whose arithmetic wraps. `Inner::checked` gives `None` where an integer result could overflow. They block for the caches, as GotoBLAS and BLIS do. Their innermost loops run on explicit vectors on AArch64, in wasm and on x86-64 with AVX2. Its decompositions of real and complex matrices, at `f32`, `f64` and `Complex` with parts of either type, are `lu`, `cholesky`, `qr`, `qr_complete` and `svd`, with `Lu::solve`, `Lu::inverse`, `Lu::lower`, `Lu::upper`, `Lu::determinant`, `Lu::log_determinant`, `Svd::solve`, `Svd::pseudoinverse`, `singular_values` and the least-squares solver `lstsq`. `solve` and `pinv` give the least-norm solution and the pseudoinverse of any matrix, singular or not. `rank` counts the singular values above `lstsq`'s cutoff. `eigh` and `hermitian_eigenvalues` take Hermitian matrices, and `eig` and `eigenvalues` any square matrix. `transpose_into` transposes a matrix through 8×8 tiles, applying a conversion to each item. `Scalar` is the sealed trait of the four item types. `linalg::exact` works in any field, such as exact rationals: `echelon`, `solve`, `pinv`, `determinant` and Ryser's `permanent`, each with a `_with` version.
- `exact`: `residue`, `gcd`, `lcm` and `binom` of exact rationals, at any integer type, such as `BigRational`. `linalg::exact` solves matrices of them.
- `distribution`: sixteen distributions. The objects `Pdf`, `Cdf` and `Quantile` hold a distribution and evaluate it over a slice. Each distribution also samples from a seeded generator.
- `random`: rolls and deals that fill a slice from one generator.
- `gather`: copying a slice's items into a buffer by position: `cycle`, `repeat`, `compress` by a mask, `take` at indices, `windows`, `walk` along a pattern of `Steps` with fills, and `scatter` at offsets, each applying the caller's conversion to each item. `walk` also takes `Same`, which copies whole slices of one type. `Steps::reach` gives the source offset of one item of a walk, and `Steps::run` the offset of a run of items that lie in order in the source.
- `convert`: `int` converts a whole float to an integer type. It gives `None` outside the type's range, and for `i64` and `u32` beyond a magnitude of 2^51 - 1. `float` converts an integer to the nearest float. Loops that call either vectorize on every target. AVX2 and wasm lack a vector instruction between `f64` and 64-bit integers. On those targets a loop that converts with `as` runs one item at a time.

The special functions `gamma`, `lgamma`, `erf`, `erfc`, their inverses, the incomplete gamma and beta functions and Lambert W have free functions and objects at the crate root, as the elementary functions do.

`Add`, `Subtract` and `Multiply` are objects at the crate root, for every float, complex and integer type. Their integer forms wrap, as NumPy's integer arrays do. `Overflowing` gives each integer type's arithmetic with whether its result fits, for callers that keep integers exact.

## Accuracy and speed

The table gives nanoseconds per item for a loop over 65,536 inputs on an Apple M5 Max, in October 2026, beside the standard library's version of each function. A row named after an object times its slice form. `Pow.one` and `Root.one` time the scalar forms. To print the table on your own machine, run `RUSTYMATH_COSTS=1 cargo test --test speed`.

| Function | f64 | std f64 | f32 | std f32 |
|---|---|---|---|---|
| `Exp` | 0.84 | 3.92 | 0.33 | 0.91 |
| `Expm1` | 1.07 | 2.70 | 0.48 | 3.10 |
| `Ln` | 1.36 | 1.62 | 0.62 | 1.34 |
| `Log1p` | 1.80 | 2.31 | 0.85 | 1.39 |
| `Log2` | 1.36 | 1.62 | 0.61 | 1.27 |
| `Log10` | 1.38 | 1.74 | 0.62 | 1.27 |
| `Sin` | 0.87 | 4.95 | 0.36 | 4.68 |
| `Cos` | 0.91 | 5.12 | 0.40 | 5.01 |
| `Tan` | 1.14 | 2.58 | 0.45 | 4.68 |
| `Asin` | 0.94 | 4.55 | 0.34 | 3.83 |
| `Acos` | 1.03 | 6.03 | 0.37 | 5.15 |
| `Atan` | 0.89 | 6.04 | 0.34 | 2.16 |
| `Sinh` | 1.26 | 2.47 | 0.56 | 2.37 |
| `Cosh` | 0.95 | 2.27 | 0.38 | 2.30 |
| `Tanh` | 1.24 | 3.44 | 0.53 | 2.28 |
| `Asinh` | 1.50 | 2.56 | 0.69 | 1.87 |
| `Acosh` | 1.59 | 2.17 | 0.68 | 1.56 |
| `Atanh` | 1.40 | 3.78 | 0.62 | 1.62 |
| `Cbrt` | 1.11 | 1.39 | 0.89 | 1.85 |
| `Pow` | 2.92 | 7.28 | 1.61 | 2.06 |
| `Pow.one` | 4.11 | 7.18 | 2.27 | 1.97 |
| `Hypot` | 0.55 | 1.21 | 0.27 | 0.87 |
| `Root` | 3.38 | 4.24 | 1.76 | 1.96 |
| `Root.one` | 4.23 | 4.25 | 2.36 | 2.09 |
| `Maximum` | 0.19 | 0.19 | 0.10 | 0.09 |
| `reduce::sum` | 0.06 | 0.49 | 0.03 | 0.46 |
| `reduce::max` | 0.06 | 0.11 | 0.03 | 0.05 |
| `reduce::cumsum` | 0.15 | 1.94 | 0.15 | 2.61 |
| `reduce::cummax` | 0.15 | 2.53 | 0.07 | 2.54 |
| `polynomial::evaluate` | 0.51 | 2.60 | 0.22 | 2.60 |

An `f16` computes in `f32` and rounds once. Special values, such as infinities, NaN and signed zeros, give IEEE's results. The tests compare each function with a correctly rounded reference at every `f16` input, at samples of `f32` and `f64` inputs, and at published hard cases.

Where the target has a fused multiply-add instruction, as on AArch64 and on x86-64 with FMA, the functions use it. Elsewhere, as in wasm, they multiply and add separately, because a call to a software `fma` would stop the loop vectorizing. `FUSED` says which the target uses. The modules `fused` and `unfused` hold every function again, with the choice made. A loop compiled with FMA for itself, such as an AVX2 version chosen when your program runs, calls `fused`. The 4-ulp bound holds either way.

At f64, `pow` and `root` compute ln |x| and its product with y as double words, through Arm Optimized Routines' tables. The result stays within 4 ulps even where y ln |x| nears the overflow threshold. A power or root whose exact result is a float, such as 15^4 or ∛27, comes out exactly. Their scalar forms take one branch for special values, as Arm's scalar `pow` does. Their slice forms compute every item the fast way, then redo only the rare items with special values. At `f32` and `f16`, `pow` and `root` compute in `f64` and round once. There the free `pow` and `root` have no branch. A loop that calls them vectorizes.
