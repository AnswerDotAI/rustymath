# Development

## Commands

```bash
cargo test
RUSTYMATH_TIMES=1 cargo test --test speed -- --nocapture
cargo develop && pytest -q
nbdev-test
nbdev-preview
ship-rs-build
```

`cargo test` runs the accuracy tests, the speed test and the doc examples, in about a second. Run plain `cargo test` while you work. `tests/speed.rs` fails when a function takes at least 1.25 times the standard library's time, or twice for `pow` and `root`. The margin allows headroom for timing noise. The special functions, which the standard library lacks, fail when they take more than their budget in copies of floats of the same width. Either failure usually means a loop stopped vectorizing. With `RUSTYMATH_TIMES` set, the test prints the README's table, then the special functions' costs.

## Layout

The root package is a pure Rust crate, published to crates.io. Rust users who depend on it never build PyO3. The Python extension is the `py` package. It depends on the root crate, is never published, and maturin builds it through `[tool.maturin].manifest-path`.

`py/src/lib.rs` makes each function a NumPy ufunc with loops for float16, float32 and float64. `ufuncs!` generates a kernel type for each function, and one generic inner loop serves them all. Aligned contiguous operands run as a slice loop that vectorizes. Other operands step through their strides. `power` runs `Pow`'s slice form on `CHUNK` items at a time, broadcast scalars included, and copies each chunk into NumPy's output. `sin`, `cos` and `tan` run the slice loop on their checked forms, then redo any item that gave `None`. The tolerant ufuncs take the tolerance as an extra input and read its first item. The wrappers in `python/rustymath/__init__.py` check it and pass it. `py/src/search.rs` exposes the tolerant searches, from which `unique`, `index` and `isin` in `__init__.py` build the results NumPy gives.

The other modules have one file each in `py/src/`. `polynomial.rs` takes contiguous arrays of one dtype, which the wrappers in `python/rustymath/polynomial.py` prepare, following the conventions of `numpy.polynomial.polynomial`. `number_theory.rs` and `distribution.rs` become the submodules `_core.number_theory` and `_core.distribution`, which `__init__.py` also registers as `rustymath.number_theory` and `rustymath.distribution`. They're submodules because the distribution constructor `gamma` and the ufunc `gamma` can't share a module. The number theory functions take Python integers through `BigUint`, and check for Ctrl-C through `Python::check_signals`.

## Documentation

The docs site is built by nbdev and Quarto from `nbs/`. `index.ipynb` is its home page, with Python examples as code cells and Rust examples as fenced blocks. `nbdev-test` runs the notebooks' code cells, and CI runs it after `pytest`. `_quarto.yml` turns execution off, so the site shows the outputs stored in each notebook. `nbdev-preview` serves the site locally, and `.github/workflows/docs.yml` deploys it to GitHub Pages on each push to `main`. `nbdev-readme` writes the README from `index.ipynb`. The Rust section and the accuracy notes live in `src/lib.md`. `lib.rs` includes that file as the crate docs, and `cargo test` runs its examples. `index.ipynb` includes it through Quarto's `include` shortcode.

## Functions

Each function is scalar Rust with no branch on its usual path, generic over `Float`. The compiler inlines it into the loops that call it, such as BPL's kernels, and vectorizes them. `lib.rs` declares every function once in `functions!`, which also puts it in the modules `fused` and `unfused`. `mla` fuses its multiply and add only where the target has FMA (`FUSED`). A software `fma` in wasm would stop the loop vectorizing. Code compiled with FMA for itself calls `fused`. `pow`, `root` and `cbrt`, `pow`'s core, and `table.rs`'s logarithm and exponential carry `#[inline(always)]`. LLVM would otherwise leave them out of line, and the loops that call them would stop vectorizing. The `tolerant` module holds APL's tolerant comparison.

LLVM vectorizes a loop that reads a table only where the target has gathers, as AVX2 does. NEON and wasm have none, so on Apple silicon such a loop runs one item at a time, even with unchecked indexing. LLVM keeps the constants that don't fit in vector registers in general registers and copies one in for each use on every iteration, so each distinct constant costs an instruction per pass. A long chain of dependent operations limits a loop as much as its instruction count does: Estrin's form for a polynomial, or z² beside b z in place of b z z z, shortens the chain. `cargo rustc --profile test --test speed -- --emit asm` writes the speed test's assembly to `target/debug/deps/speed-*.s`, one file per codegen unit.

At f64, `pow` and `root` use `table.rs`: Arm Optimized Routines' logarithm and exponential, with tables of 1/c, ln c and 2^(j/256), and double words (`dd.rs`). At f32 and f16 they compute e^(y ln |x|) in f64 with no table, so that their loops vectorize everywhere. `ln_narrow` and `exp_narrow` serve those widths, and the special functions' narrow paths, with polynomials that keep only the accuracy a narrower result needs. `pow` and `root` handle special values at the argument's own width, after the result narrows. Testing them at f64 converted each mask between f64 and f32 lanes and cost f32 `pow` about a quarter of its time. `cbrt` takes its first estimate from the bits of x and needs no table.

`src/accuracy.rs` checks every function, with `mla` fused and unfused, at every f16 input, at samples of f32 and f64 inputs and at the cases in `testdata/`. Every check must be within `BOUND`, 4 ulps. The elementary functions are checked against the platform's libm through Rust's standard library, except `asinh`, `acosh` and `atanh`, which the standard library computes with formulas of its own and the tests call from libm directly. Rounded to f32 or f16, libm's f64 result is a correctly rounded reference. At f64 a check against libm allows `LIBM`, 2 ulps more, for libm's own error. The hard cases in `testdata/core-math.txt` and the special functions' cases in `testdata/mpmath.txt` carry correctly rounded results from mpmath, which `python testdata/references.py` writes. The special functions' f16 checks round rustymath's own f64 result, and the inverse incomplete functions recover each argument from its probability in `mpmath.txt`. The ported algorithms' notices are in `THIRD-PARTY.md`. The test data files carry their own. `links/`, which git ignores, holds shallow clones of the projects that tests and algorithms come from.

## Slice forms

`slice.rs` gives a function a second form for whole slices. A `functions!` entry marked `as`, such as `pow(x, y) in pow as Pow`, declares a function object. The object's `one` is the scalar function. Its `many` takes each argument as `Arg::One`, an item that pairs with every result, or `Arg::Many`, a slice. It appends the results to a `Vec` and gives `false` when some result is NaN where neither argument is. `checked` is `one` with the same rule. rustymath gives NaN for an argument outside a function's real domain. A caller such as basedpl then takes another path for the whole call.

The object's forms at f32 and f64 are concrete functions in rustymath, which the sealed trait `BinaryForms` holds. `Dispatch` picks the form for each float type. At f16, `many` converts `CHUNK` items at a time to f32 and rounds the results. No caller inlines these forms. A caller's build settings therefore can't change their loops. An edit to a function's body rebuilds only rustymath. basedpl then relinks, in about 2 seconds for `pow`.

A module's slice forms live in its `slices` submodule. `pow`'s runs `two_pass`. Every item takes `pow_plain`, and a second pass recomputes with `pow` only the rare items that need its special cases. At f32 this beats libm, where the scalar `pow` doesn't.

`pairs` is the loop over the pairings of items and slices. It writes into the result's spare capacity without zeroing it. The slice forms use it, and so does `Arg::converted`. A caller's own kernels can use it too, as basedpl's arithmetic does. `CHUNK` is how many items a caller converts or expands into buffers at a time.

## Number theory

`number_theory.rs` works on integers, so it doesn't use `functions!`. `Natural` is sealed by `Tier`, which gives each type its next narrower type: `u128` for `BigUint` and `u64` for `u128`. Each public function first moves its argument to the narrowest type that holds it, so small `BigUint` values compute at `u64` speed. `u64` and `u128` compute in Montgomery form, which `montgomery!` generates. `BigUint` uses plain residues and `modpow`. A narrowing call passes `check` on by value. Passing `&mut check` instead gives each level a new closure type, and the compiler never finishes instantiating the `u64` case. `Primes` keeps one byte for each 30 numbers, with a bit for each residue coprime to 30. It starts each segment from a copied pattern of the multiples of 7, and its segments double from 128 bytes to 32 KiB. The module's tests take a few hundredths of a second.

## Polynomials

`polynomial.rs` holds slice functions, so it doesn't use `functions!` either. `evaluate` runs Horner's rule on fixed blocks of 32 items from `as_chunks_mut`, and pads the last partial block. With a variable length, a caller built with many codegen units, such as basedpl, calls `memcpy` and `Zip::new` once per block, and evaluation costs about three times as much. rustymath's speed test calls `evaluate` from a small crate of its own. Nobody has checked whether it shows this cost at basedpl's settings. Time a change in basedpl too. `roots` and `fit` call `faer` in `f64`.

## Search

`search.rs` and the slice functions in `tolerant.rs` came from basedpl, whose search and grade call them. A slice holds at most `u32::MAX` items, because positions take 32 bits inside them. `index_of`, `classes` and `firsts` index keys in a table or a `foldhash` map. A tolerant search with a nonzero tolerance files floats in `Near`'s buckets, and tolerant classification splits sorted order keys into runs. `Near` takes its bucket width from the tolerance and the float's epsilon, and a test compares it with a brute-force search at each width and at small and large tolerances. Time changes in basedpl, whose speed test has the search and grade cases.

## Versioning

The canonical version lives in `[workspace.package]` in `Cargo.toml`, which both crates inherit. `pyproject.toml` gets the Python package version from Cargo via `dynamic = ["version"]`.

## Build profiles

The profiles match basedpl's line for line. Cargo ignores a dependency's own profiles. Inside basedpl, rustymath compiles at basedpl's settings. With the same settings here, the speed test times code compiled the way basedpl compiles it. Change both `Cargo.toml` files together.

`release` compiles at opt-level 2, incrementally and without LTO, into Cargo's default of 256 codegen units. `cargo test` and `cargo develop` share the `test` profile, which inherits `release`. `cargo develop` installs the extension from that build into the editable package. Editable installs through `uv sync` or `pip install -e` compile no Rust, because the build backend is `fastws.build_backend`. In CI, `cargo develop` runs after `cargo test` in the same job and reuses its build. Don't use `maturin develop`. It compiles a second copy of the crate, unoptimized unless given `--release`.

CI builds distributed wheels with `dist`. It turns incremental compilation off and strips the result. Without incremental compilation, Cargo uses 16 codegen units, with thin LTO across them.

## Release

1. Run `cargo develop && pytest -q`.
2. Confirm the release version in `Cargo.toml` (`[workspace.package].version`).
3. Run `ship-release`.

`ship-release` pushes a `v<version>` tag, which starts the release workflow in GitHub Actions, then bumps the version.
