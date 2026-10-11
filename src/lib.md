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

### Tolerant comparison

The `tolerant` module has APL's tolerant comparison: x equals y when |x - y| is at most the tolerance times the larger of |x| and |y|. Its objects, such as `Equal`, `Less`, `Floor`, `Residue` and `Gcd`, each hold a `Tolerance`. With `Tolerance::EXACT`, a slice form runs a loop of plain IEEE comparisons, chosen once per call.

```rust
use rustymath::{tolerant::{Equal, Tolerance}, Arg, Binary};

let mut same = Vec::new();
Equal(Tolerance::DEFAULT).many(Arg::Many(&[0.3, 0.1 + 0.2, 0.31]), Arg::One(0.3), 3, &mut same); // same: [true, true, false]
```

### Other modules

- `number_theory`: primality, factorisation, next and previous primes, Euler's totient, and the enumeration and counting of primes, on a `u64`, a `u128` or a `BigUint`. Each computes in the narrowest of those that holds the value. A function that can run for a long time also has a `_with` version, whose extra argument is a check that can stop it.
- `polynomial`: evaluation over a slice by Horner's rule, roots from the eigenvalues of the companion matrix, building from roots, derivatives and least-squares fitting.
- `reduce`: sums, products, maximums and minimums over a slice, and their running versions. Float sums and products vectorize by combining the items in any order.
- `distribution`: sixteen distributions. The objects `Pdf`, `Cdf` and `Quantile` hold a distribution and evaluate it over a slice. Each distribution also samples from a seeded generator.
- `random`: rolls and deals that fill a slice from one generator.

The special functions `gamma`, `lgamma`, `erf`, `erfc`, their inverses, the incomplete gamma and beta functions and Lambert W have free functions and objects at the crate root, as the elementary functions do.

## Accuracy and speed

The table gives nanoseconds per item for a loop over 65,536 inputs on an Apple M5 Max, in October 2026, beside the standard library's version of each function. A row named after an object times its slice form. `Pow.one` and `Root.one` time the scalar forms. To print the table on your own machine, run `RUSTYMATH_TIMES=1 cargo test --test speed -- --nocapture`.

| Function | f64 | std f64 | f32 | std f32 |
|---|---|---|---|---|
| `Exp` | 0.98 | 1.67 | 0.40 | 0.98 |
| `Expm1` | 1.16 | 2.03 | 0.51 | 1.83 |
| `Ln` | 1.41 | 1.52 | 0.61 | 1.30 |
| `Log1p` | 1.87 | 2.17 | 0.83 | 1.30 |
| `Log2` | 1.39 | 1.52 | 0.61 | 1.19 |
| `Log10` | 1.41 | 1.63 | 0.61 | 1.19 |
| `Sin` | 0.88 | 2.23 | 0.39 | 1.57 |
| `Cos` | 1.00 | 2.19 | 0.44 | 1.66 |
| `Tan` | 1.21 | 2.34 | 0.49 | 1.93 |
| `Asin` | 1.00 | 2.43 | 0.34 | 1.30 |
| `Acos` | 1.08 | 2.26 | 0.40 | 1.61 |
| `Atan` | 1.04 | 2.57 | 0.41 | 2.19 |
| `Sinh` | 1.34 | 2.32 | 0.57 | 1.47 |
| `Cosh` | 1.08 | 2.28 | 0.44 | 1.69 |
| `Tanh` | 1.32 | 2.16 | 0.57 | 1.65 |
| `Asinh` | 1.67 | 2.40 | 0.71 | 1.76 |
| `Acosh` | 1.63 | 2.17 | 0.71 | 1.75 |
| `Atanh` | 1.44 | 2.34 | 0.61 | 1.52 |
| `Cbrt` | 1.22 | 1.74 | 0.88 | 1.74 |
| `Pow` | 3.29 | 4.19 | 1.86 | 1.97 |
| `Pow.one` | 3.93 | 4.19 | 2.21 | 1.94 |
| `Hypot` | 0.75 | 1.09 | 0.36 | 0.87 |
| `Root` | 3.42 | 3.97 | 1.92 | 1.96 |
| `Root.one` | 4.12 | 4.01 | 2.17 | 2.00 |
| `reduce::sum` | 0.06 | 0.47 | 0.03 | 0.47 |
| `reduce::max` | 0.08 | 0.11 | 0.04 | 0.06 |
| `reduce::cumsum` | 0.15 | 2.61 | 0.15 | 2.61 |
| `polynomial::evaluate` | 0.48 | 3.04 | 0.21 | 2.93 |

An `f16` computes in `f32` and rounds once. Special values, such as infinities, NaN and signed zeros, give IEEE's results. The tests compare each function with a correctly rounded reference at every `f16` input, at samples of `f32` and `f64` inputs, and at published hard cases.

Where the target has a fused multiply-add instruction, as on AArch64 and on x86-64 with FMA, the functions use it. Elsewhere, as in wasm, they multiply and add separately, because a call to a software `fma` would stop the loop vectorizing. `FUSED` says which the target uses. The modules `fused` and `unfused` hold every function again, with the choice made. A loop compiled with FMA for itself, such as an AVX2 version chosen when your program runs, calls `fused`. The 4-ulp bound holds either way.

At f64, `pow` and `root` compute ln |x| and its product with y as double words, through Arm Optimized Routines' tables. The result stays within 4 ulps even where y ln |x| nears the overflow threshold. A power or root whose exact result is a float, such as 15^4 or ∛27, comes out exactly. Their scalar forms take one branch for special values, as Arm's scalar `pow` does. Their slice forms compute every item the fast way, then redo only the rare items with special values. At `f32` and `f16`, `pow` and `root` compute in `f64` and round once. There the free `pow` and `root` have no branch. A loop that calls them vectorizes.
