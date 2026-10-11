# Development

## Commands

```bash
cargo test
RUSTYMATH_TIMES=1 cargo test --test speed -- --nocapture
maturin develop && pytest -q
nbdev-test
nbdev-preview
ship-rs-build
```

`cargo test` runs the accuracy tests, the speed test and the doc examples. `tests/speed.rs` fails when a function is slower than the standard library's version, or when `pow` or `root` takes more than twice its time. The special functions, which the standard library lacks, fail when they take more than their budget in copies of floats of the same width, about twice their cost on the Mac. Either failure usually means a loop stopped vectorizing. With `RUSTYMATH_TIMES` set, the test prints the README's table, then the special functions' costs.

## Layout

The root package is a pure Rust crate, published to crates.io. Rust users who depend on it never build PyO3. The Python extension is the `py` package. It depends on the root crate, is never published, and maturin builds it through `[tool.maturin].manifest-path`.

`py/src/lib.rs` makes each function a NumPy ufunc with loops for float16, float32 and float64. `ufuncs!` generates a kernel type for each function, and one generic inner loop serves them all. Aligned contiguous operands run as a slice loop that vectorizes. Other operands step through their strides. `sin`, `cos` and `tan` run the slice loop on their checked forms, then redo any item that gave `None`. The tolerant ufuncs take the tolerance as an extra input and read its first item. The wrappers in `python/rustymath/__init__.py` check it and pass it. `py/src/search.rs` exposes the tolerant searches, from which `unique`, `index` and `isin` in `__init__.py` build the results NumPy gives.

The other modules have one file each in `py/src/`. `polynomial.rs` takes contiguous arrays of one dtype, which the wrappers in `python/rustymath/polynomial.py` prepare, following the conventions of `numpy.polynomial.polynomial`. `number_theory.rs` and `distribution.rs` become the submodules `_core.number_theory` and `_core.distribution`, which `__init__.py` also registers as `rustymath.number_theory` and `rustymath.distribution`. They're submodules because the distribution constructor `gamma` and the ufunc `gamma` can't share a module. The number theory functions take Python integers through `BigUint`, and check for Ctrl-C through `Python::check_signals`.

## Documentation

The docs site is built by nbdev and Quarto from `nbs/`. `index.ipynb` is its home page, with Python examples as code cells and Rust examples as fenced blocks. `nbdev-test` runs the notebooks' code cells, and CI runs it after `pytest`. `_quarto.yml` turns execution off, so the site shows the outputs stored in each notebook. `nbdev-preview` serves the site locally, and `.github/workflows/docs.yml` deploys it to GitHub Pages on each push to `main`. `nbdev-readme` writes the README from `index.ipynb`. The Rust section and the accuracy notes live in `src/lib.md`. `lib.rs` includes that file as the crate docs, and `cargo test` runs its examples. `index.ipynb` includes it through Quarto's `include` shortcode.

## Functions

Each function is scalar Rust with no branch on its usual path, generic over `Float`. The compiler inlines it into the loops that call it, such as BPL's kernels, and vectorizes them. `lib.rs` declares every function once in `functions!`, which also puts it in the modules `fused` and `unfused`. `mla` fuses its multiply and add only where the target has FMA (`FUSED`). A software `fma` in wasm would stop the loop vectorizing. Code compiled with FMA for itself calls `fused`. `pow`, `root` and `cbrt`, `pow`'s core, and `table.rs`'s logarithm and exponential carry `#[inline(always)]`. LLVM would otherwise leave them out of line, and the loops that call them would stop vectorizing. The `tolerant` module holds APL's tolerant comparison.

LLVM vectorizes a loop that reads a table only where the target has gathers, as AVX2 does. NEON and wasm have none, so on Apple silicon such a loop runs one item at a time, even with unchecked indexing. LLVM keeps the constants that don't fit in vector registers in general registers and copies one in for each use on every iteration, so each distinct constant costs an instruction per pass. A long chain of dependent operations limits a loop as much as its instruction count does: Estrin's form for a polynomial, or z² beside b z in place of b z z z, shortens the chain. `cargo rustc --profile test --test speed -- --emit asm` writes the speed test's assembly to `target/debug/deps/speed-*.s`, one file per codegen unit.

At f64, `pow` and `root` use `table.rs`: Arm Optimized Routines' logarithm and exponential, with tables of 1/c, ln c and 2^(j/256), and double words (`dd.rs`). At f32 and f16 they compute 2^(y log2 |x|) in plain f64 with no table, so that their loops vectorize everywhere. `cbrt` takes its first estimate from the bits of x and needs no table.

`src/accuracy.rs` checks every function, with `mla` fused and unfused, at every f16 input, at samples of f32 and f64 inputs and at the cases in `testdata/`. Every check must be within `BOUND`, 4 ulps. The elementary functions are checked against the platform's libm through Rust's standard library, except `asinh`, `acosh` and `atanh`, which the standard library computes with formulas of its own and the tests call from libm directly. Rounded to f32 or f16, libm's f64 result is a correctly rounded reference. At f64 a check against libm allows `LIBM`, 2 ulps more, for libm's own error. The hard cases in `testdata/core-math.txt` and the special functions' cases in `testdata/mpmath.txt` carry correctly rounded results from mpmath, which `python testdata/references.py` writes. The special functions' f16 checks round rustymath's own f64 result, and the inverse incomplete functions recover each argument from its probability in `mpmath.txt`. The ported algorithms' notices are in `THIRD-PARTY.md`. The test data files carry their own. `links/`, which git ignores, holds shallow clones of the projects that tests and algorithms come from.

## Number theory

`number_theory.rs` works on integers, so it doesn't use `functions!`. `Natural` is sealed by `Tier`, which gives each type its next narrower type: `u128` for `BigUint` and `u64` for `u128`. Each public function first moves its argument to the narrowest type that holds it, so small `BigUint` values compute at `u64` speed. `u64` and `u128` compute in Montgomery form, which `montgomery!` generates. `BigUint` uses plain residues and `modpow`. A narrowing call passes `check` on by value. Passing `&mut check` instead gives each level a new closure type, and the compiler never finishes instantiating the `u64` case. `Primes` keeps one byte for each 30 numbers, with a bit for each residue coprime to 30. It starts each segment from a copied pattern of the multiples of 7, and its segments double from 128 bytes to 32 KiB. The module's tests take a few hundredths of a second.

## Polynomials

`polynomial.rs` holds slice functions, so it doesn't use `functions!` either. `evaluate` runs Horner's rule on fixed blocks of 32 items from `as_chunks_mut`, and pads the last partial block. With a variable length, a caller built with many codegen units, such as basedpl, calls `memcpy` and `Zip::new` once per block, and evaluation costs about three times as much. rustymath's own speed test can't show this, because its small test crate inlines both. Time a change in basedpl too. `roots` and `fit` call `faer` in `f64`.

## Search

`search.rs` and the slice functions in `tolerant.rs` came from basedpl, whose search and grade call them. A slice holds at most `u32::MAX` items, because positions take 32 bits inside them. `index_of`, `classes` and `firsts` index keys in a table or a `foldhash` map. A tolerant search with a nonzero tolerance files floats in `Near`'s buckets, and tolerant classification splits sorted order keys into runs. `Near` takes its bucket width from the tolerance and the float's epsilon, and a test compares it with a brute-force search at each width and at small and large tolerances. Time changes in basedpl, whose speed test has the search and grade cases.

## Versioning

The canonical version lives in `[workspace.package]` in `Cargo.toml`, which both crates inherit. `pyproject.toml` gets the Python package version from Cargo via `dynamic = ["version"]`.

## Build profiles

uv builds and `maturin develop --release` use the incremental `release` profile for fast local iteration. CI builds distributed wheels with `dist`, which enables full LTO and one codegen unit, disables incremental compilation, and strips the result.

`cargo test` uses the `test` profile. It inherits `release`, because the speed test must time the vectorized code that callers get.

## Release

1. Run `maturin develop && pytest -q`.
2. Confirm the release version in `Cargo.toml` (`[workspace.package].version`).
3. Run `ship-release`.

`ship-release` pushes a `v<version>` tag, which starts the release workflow in GitHub Actions, then bumps the version.
