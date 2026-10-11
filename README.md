# rustymath

rustymath has fast maths functions for `f32`, `f64` and `half::f16`, written in plain Rust. Each function is a scalar function. Where it has no branches, the compiler inlines it into the loop that calls it and vectorizes that loop, as it would for `+` or `×`. rustymath gives up the last bit of accuracy for speed. Each function documents its error bound in units in the last place (ulps).

rustymath is at an early stage. It has `exp`, `ln`, `sin` and `cos`, and APL's tolerant comparison.

## Install

Add rustymath from crates.io to your `Cargo.toml`:

```toml
[dependencies]
rustymath = "0.1"
```

## Example

```rust
use rustymath::{checked_sin, exp, ln};

let xs = [0.5f32, 1.0, 2.0];
let ys: Vec<f32> = xs.iter().map(|&x| exp(x)).collect();
assert!((ln(ys[2]) - 2.0).abs() < 1e-6);
assert_eq!(ln(1.0f64), 0.0);
let sines: Option<Vec<f64>> = [0.5, 1e6].iter().map(|&x| checked_sin(x)).collect();
assert!(sines.is_some());
```

## Accuracy and speed

| Function | Width | Bound | ns per item | macOS library ns per item |
|---|---|---|---|---|
| `exp` | f64 | 1 ulp | 1.08 | 1.36 |
| `exp` | f32 | 2 ulp | 0.38 | 0.92 |
| `ln` | f64 | 2 ulp | 1.28 | 1.63 |
| `ln` | f32 | 3 ulp | 0.58 | 1.36 |
| `sin` | f64 | 4 ulp | 0.71 | 1.61 |
| `sin` | f32 | 2 ulp | 0.30 | 1.12 |
| `cos` | f64 | 4 ulp | 0.86 | 1.66 |
| `cos` | f32 | 4 ulp | 0.38 | 1.18 |

The times are for a loop over 65,536 items on an Apple M5 Max, in October 2026. The loops for sine and cosine call `checked_sin` and `checked_cos`, on arguments from -10 to 10.

An `f16` computes in `f32` and rounds once, which gives a result within 1 ulp. The special values give IEEE's results. Each function matches a correctly rounded reference within its bound at every `f32` and `f16` input, and at millions of `f64` inputs, including published hard cases.

Where the target has a fused multiply-add instruction, as on AArch64 and on x86-64 with FMA, the functions use it. Elsewhere, as in wasm, they multiply and add separately, because a call to a software `fma` would stop the loop vectorizing. `FUSED` says which the target uses. The bounds hold both ways.

## Sine and cosine

`sin` and `cos` reduce their argument by multiples of π. Up to a magnitude of 2^23 at f64 or 2^20 at f32, a few multiply-adds do it. Beyond that, an exact reduction in integer arithmetic gives the same accuracy for every finite argument. That reduction needs a branch, which stops a loop that calls `sin` or `cos` vectorizing. `checked_sin` and `checked_cos` have no branch. They give `None` beyond those magnitudes, and a loop that calls them vectorizes.

## Tolerant comparison

The `tolerant` module compares floats as APL does: `x` equals `y` within a relative tolerance `t` when `|x-y|` is at most `t` times the larger of `|x|` and `|y|`. Its tolerances give tolerant equality, order, floor, ceiling, residue, whole numbers and greatest common divisors, and the range of floats equal to a number. When a tolerance is 0, `with_tolerance!` picks `Exactly`, whose methods compile to plain IEEE comparisons.

```rust
use rustymath::tolerant::{Tolerance, Tolerant};

let t = Tolerance::DEFAULT;
assert!(t.equal(0.1 + 0.2, 0.3));
assert_eq!(t.floor(2.9999999999999996), 3.0);
```

## Licence

Apache-2.0. Some algorithms and constants come from SLEEF and from Arm Optimized Routines. `THIRD-PARTY.md` has their notices.
