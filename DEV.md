# Development

Rustymath favours speed and a small implementation over last-bit agreement with other libraries. It doesn't aim for reproducible results either. The last bits of a result can differ between thread counts, platforms, builds and runs. Never spend speed, memory or code to make them agree. Tests never compare results across thread counts. Keep the accuracy contracts that callers use. Check a change that alters arithmetic against those contracts, not against bit-identical results. Don't add sign canonicalization or compensated arithmetic without an agreed need.

## Commands and builds

```bash
cargo test
RUSTYMATH_COSTS=1 cargo test --test speed -- --nocapture
cargo develop && pytest -q
nbdev-test
nbdev-preview
```

Run plain `cargo test` while you work. It runs the Rust integration tests, speed test and doctests. `cargo develop` builds the same `test` profile and installs the Python extension into the editable package. Rebuild before testing Python changes that depend on Rust. An already-running Python process retains its loaded extension.

Build nothing else without explicit discussion: no scratch crates, alternate target directories, `--release` or `-p` builds. The exception for inspecting x86-64 machine code is described in basedpl's `DEV.md`, under “X86-64 machine code”. Use that procedure rather than `--emit asm`, which changes the codegen-unit arrangement you are inspecting. Packaging checks use `ship-rs-build`; pass `--profile dist` when checking a distributed artifact.

The root package is the Python-free crate published to crates.io. `py/` is the unpublished PyO3 extension. `pyproject.toml` points maturin at its manifest. Editable installation through `uv sync` or `pip install -e` uses `fastws.build_backend` and compiles no Rust. Run `cargo develop` after installation. Don't use `maturin develop`: its build configuration differs from the shared local build.

The `test` profile inherits `release`: optimization level 2, incremental compilation, LTO off, no debug information, overflow checks or debug assertions. The configs explicitly enable LLVM's SLP vectorizer. On x86-64 they also target x86-64-v3 for AVX2 and FMA. Without those target features, some conversions and rounding operations run one item at a time. `dist` disables incremental compilation, enables Cargo's default within-crate LTO and strips symbols.

Keep these profiles aligned with basedpl's. Cargo uses the consuming workspace's profiles for dependencies. Concrete function forms compile in rustymath, but their build still uses that workspace's profile and target flags. LTO-off incremental builds limit how much code an edit recompiles; debug line tables and inlining across units increase that work.

### Speed measurements

`tests/speed.txt` holds budgets in copies of the same number of items. The speed test times each copy immediately before its operation. The standard-library comparisons are informational. A budget failure often identifies a loop that stopped vectorizing. `RUSTYMATH_COSTS` prints every cost and the README's comparison table.

The manually triggered `Speed` workflow measures the Ubicloud runner used by `Test` and restores its cache. After a run, use `basedpl.bpltests` to set budgets under the rule in `tests/speed.txt`:

```python
set_budgets(native=[cargo_costs(rms), logged_costs(log)], run=run, path=rms/'tests/speed.txt')
```

Here `rms` is this repo's `Path`, `log` is the job log and `run` is its run number. Budget measurements use `parallel::set_threads(1)`; multicore scaling is a separate measurement.

The speed test and basedpl's timing tools take `/tmp/aai-ws-timing.lock` exclusively. Cooperating build wrappers (`cargo develop`, `cargo wasm`) and basedpl's reference/pytest tests take it shared. Bare Cargo commands and other test binaries do not all take it. After five minutes a timing fails; a build proceeds without the lock. Never hold this lock in the shared kernel process or coordinate timings manually. macOS timings request user-interactive QoS to favour performance cores; this is not CPU affinity.

## Code map

| Files | Responsibility |
|---|---|
| `lib.rs`, `float.rs` | Public functions and objects; `Float`, `Native`, fused/unfused arithmetic and sealed implementation traits. |
| `exp.rs`, `ln.rs`, `trig.rs`, `arc.rs`, `hyperbolic.rs`, `pow.rs`, `parts.rs`, `sqrts.rs` | Elementary scalar functions and specialized forms. |
| `table.rs`, `dd.rs` | Table-based f64 logarithms/exponentials and double-word arithmetic. |
| `special.rs`, `lambert.rs`, `complex.rs` | Special functions and complex forms of the same objects. |
| `slice.rs`, `parallel.rs` | Scalar/slice object interfaces, output storage and elementwise scheduling. |
| `reduce.rs` | Folds, scans and axis traversal. |
| `arith.rs`, `integer.rs`, `convert.rs` | Wrapped/checked arithmetic, integer functions and vectorizable conversions. |
| `distribution.rs`, `random.rs` | Checked distributions, function objects, sampling, rolls and deals. |
| `number_theory.rs` | Primality, factorization, prime enumeration/counting and factorial. |
| `field.rs`, `exact.rs`, `linalg/exact.rs`, `polynomial.rs` | Generic field algorithms, rational functions, exact matrices and polynomials. |
| `search.rs`, `tolerant.rs`, `gather.rs` | Exact/tolerant search and classification, stable sorting, comparisons and copying through conversions. |
| `linalg.rs`, `linalg/scalar.rs`, `linalg/factor/` | Packed products, scalar operations and decompositions. |
| `py/src/`, `python/rustymath/` | NumPy adapters and Python wrappers. |

## Functions and slice forms

Free functions are generic over `Float`: f16 computes in f32 and rounds its result, while f32 and f64 use their own working type. `Native` names f32 and f64. `functions!` declares the free wrappers at the crate root and in `fused`/`unfused`; an `as` entry also declares the function object. `FUSED` selects hardware FMA where available. Don't introduce software FMA into targets without it.

Objects implement `Unary<T>`, `Binary<T>` or `Ternary<T>` at concrete item types. They can hold settings. `one` computes a scalar, `checked` also reports failure, and `many` appends results to a vector and returns whether every item succeeded. Arguments are `Arg::One` or `Arg::Many`. Generic code calling an object states its bound, such as `Exp: Unary<F, Out = F>`; free functions need no object bound.

`forms!` writes the sealed `Forms<E, N>` implementations. `at real` writes f32/f64 forms and an f16 adapter through f32. Explicit type lists cover complex, integer and Boolean forms. `=> T` names an output type different from the input. The item body returns one of three outcomes:

- A plain float/complex value fails on NaN from arguments that aren't NaN. Plain integer and Boolean values have no NaN failure.
- `Option` fails on `None`.
- `Total` never fails, including where NaN is the intended result.

Arithmetic objects deliberately accept IEEE NaNs and wrapped integer results. Integer functions that need to report overflow return `Option`. A `: total` entry in `functions!` wraps the body in `Total`. Test an object's actual verdict rather than applying the ordinary NaN rule to every object.

`arities!` gives the public traits one implementation through `Forms`. Keep objects concrete: their forms compile in rustymath instead of being instantiated in every caller. Forms that differ from the free functions, including `Sin`, `Cos`, `Tan`, `Pow` and `Root`, live beside their implementations.

### Loops and codegen

`Forms::slice` writes into provided slots. `many` and `pairs` reserve spare capacity through `appended`, without zeroing it. `written` views initialized slots for later passes. Keep that unsafe storage handling in these helpers. `many_into` and `Slots::pairs` run the same loops into a `parallel::Slots`, which `extended` and `Slots::append` hand out. A caller such as basedpl's fused chains then writes its last step straight into its result. Each writer fills every slot it gets, and a writer that doesn't panics before the vector's length changes.

`each` runs the ordinary item loop. It checks outputs first and inspects arguments only for NaN outputs. `two_pass` computes a branch-free fast path, then recomputes its rare misses. `split` separates complex parts into real buffers, calls real slice forms and redoes exceptional items with the scalar function. `CHUNK` governs the cache-sized pieces used by these forms and their adapters.

Keep these performance rules when changing loops:

- Pass item bodies through `each!`, `two_pass!` or `split!`. They mark closures `#[inline(always)]`; leaving a large body out of line can stop vectorization.
- Write separate closures for functions selected at runtime. A shared closure can leave the selector inside every iteration. `trig.rs` shows the pattern.
- Keep `pairs` at `#[inline(never)]`. Inlining a function whose loop runs behind `parallel::split`'s `&dyn` only repeats its setup at every call site.
- Keep generic loop helpers inlinable when their instances need to compile in the caller's codegen unit. This includes nested helpers. Basedpl's `tests/units.rs` checks the resulting split.
- Check code in basedpl as well as in this crate when changing a generic function that it instantiates. The small test crate is not the same compilation context.
- Fill a large per-call buffer through `appended`'s spare capacity, or with `Vec::with_capacity` and `resize` as `linalg`'s `filled` does. On macOS, `vec![0; n]` allocates through calloc. Calloc's `madvise` costs more than writing the items. Its cost grows with the number of threads that touch the buffer.

Elementary kernels aim for branch-free vectorizable loops. Incomplete gamma/beta and their inverses also use data-dependent series, continued fractions and iterations, with precision-appropriate stopping. Table gathers help AVX2 but are unavailable in NEON/wasm. Short dependency chains and vectorizable conversions matter as much as instruction count.

`Pow`/`Root` use table-based f64 arithmetic and narrow-width vector paths. Their scalar object forms have separate fast paths; f32 scalar calls use tables that the vector paths avoid. A whole exponent from 2 to 8 squares and multiplies, in the scalar form and in a slice form with a constant exponent. A degree of 2 or 3 takes `sqrt` or `cbrt`. A table-free f64 `Pow` measured about 30% slower: with `ln` and `exp` in one loop, their polynomial constants don't fit in registers. The loop then rebuilds them for every item. `cbrt` computes at each width's own precision. Trig slice forms redo large arguments after their fast reduction. f64 inverse-error slice forms group inputs by approximation region. `Erfc` chooses a single formula for blocks wholly on one side of its split. Normal/log-normal CDF and quantile objects reuse those optimized forms through `composed`.

### Parallel execution

`parallel::ranges` schedules independent ranges. `split` applies it to disjoint output slices; `many`, `pairs` and `extended` use that interface. A long call times its first unit, then cuts the rest into pieces of at least `PIECE_TIME` of work, at most four for each thread. `pieces` runs fewer pieces than threads on a pool of the largest power of two up to their number: memory-bound calls of a million items ran faster on small pools than on all 18 threads of an M5 Max. Chunked forms keep their temporary buffers in cache. Work and result initialization finish for every piece even if another piece reports failure. `parallel::extended` lets a caller such as basedpl's fusion fill a vector's spare capacity in pieces, through `Slots`.

`parallel::stack_with` runs a function on each cell of a stack of matrices, for basedpl's `f⍤2` and the binding's gufunc loops. It times its first cells on the calling thread, then splits the rest across threads in rounds of about `ROUND_TIME` for each thread, with the caller's `check` between rounds and one scratch value, such as an `LuWork`, for each piece. A cell on another thread doesn't split again.

The default pool follows `RAYON_NUM_THREADS`, otherwise Rayon's available-core default. `parallel::set_threads(n)` changes calls made on the current thread and returns the previous setting. Use 0 to restore the default or 1 for serial calls. Calls already on a Rayon worker do not split again. Wasm runs serially. Pools are created on demand and rebuilt in a child after `fork`.

This scheduler covers elementwise forms, folds and matrix products. Scans don't split. `reduce::along` folds a lane, or a group of `inner` lanes, that fits in one piece of `reduce::fold_pieces` whole, and runs of them split across threads. Larger arrays fold in one piece per thread, each of at least 2^18 items and 256 rows. A lane that crosses pieces merges its parts' folds. Splitting a single matrix's columns across threads made `+⌿` twice as slow, because each thread read part of every row. A product splits its rows into bands of about `BAND_WORK` multiply-adds, each with its own packed block of `a`. It splits once for each panel of `b` of up to `PANEL` items, or `SERIAL_PANEL` bytes when it runs on one thread, across at most √(work / `THREAD_WORK`) threads. Each thread it wakes costs about as much as `THREAD_WORK` multiply-adds. Decompositions split in the products of their blocked updates. The Hermitian and bidiagonal reductions also cut each pass's rows into one piece per thread, each of at least `ROW_PIECE` rows, through `eigen::summed_rows`. Divide and conquer splits each merge's secular roots, corrected z and vector rows. Panels and the recursion's two halves run on one thread. Callers that submit only small converted chunks also limit the work available to it. The shared non-generic scheduler keeps Rayon's machinery out of each kernel instantiation. Item types and forms are `Send + Sync` through the sealed traits.

## Other numerical modules

### Reductions

`Reduction` defines the unit, scalar step, checked step, contiguous fold, merge and scan for `Sum`, `Product`, `Max` and `Min`. Integer sums/products wrap and report whether they fit. f16 sums and products have an f32 `Total` on every fold path, column folds and `fold_pieces` merges included. The caller rounds it once. f16 extrema total in f16. Float sums/products can regroup algebraic operations. Beside NaN, float extrema follow each target's instructions. AArch64's `fmax` and `fmin` give NaN. Every other step passes the running result as `maximum`'s or `minimum`'s `y`, which a NaN item leaves unchanged.

`slice::running` chooses the partial-total count for the target and item size. Explicit vector extrema are used where they outperform ordinary loops. Wasm uses `max`/`min`, which keep NaN. They take about eight instructions each on x86-64 hosts, but run faster than `pmax`/`pmin` in a browser on an Arm host, such as a Mac. Integer extrema scans skip blocks that cannot change their running result.

`Logical` represents any Boolean dyad as a truth table and scans by composing its two-state maps. `Xor`/`Xnor` use that path too. Lanes of at most 64 consecutive items scan in one call of `mapped`, which restarts each lane inside its word. `SumAs<C>` counts Booleans or sums integers in the caller's chosen total type. The caller must choose a width that holds the totals.

`Alternating` implements right folds by subtraction or division. Float/complex pairs sum differences or multiply ratios. Integer subtraction keeps separate totals for even and odd positions. A floating quotient reports failure when its intermediate products or result leave the normal range; basedpl then folds item by item.

`along`, `scan_along` and `fold_into` traverse `[outer, len, inner]` arrays. Contiguous lanes of at least `SHORT_LANE` items use slice reductions, shorter ones fold item by item, and other lanes update a row of totals. `fold_into` accepts blocks from callers such as fusion. `extremes` and `extremes_of` compute both bounds in one pass.

### Complex functions and distributions

Complex forms attach to the same objects as real forms. `Abs` and `Angle` return real values; real `Cis` returns complex values. Most expensive complex slice forms split into vectorized real operations. `Abs`, `Angle` and `Sqrt` keep direct loops. Exceptional magnitudes and branch-cut cases take scalar paths. Inverse circle/hyperbolic functions use Kahan's formulas; complex ln Gamma uses Stirling, recurrence and reflection with SciPy's branch convention.

Complex errors are measured against a magnitude-based scale, with function-specific scales for ln Gamma, powers, Gamma and binomial. Quarter-turn directions are deliberately exact. See `complex.rs` for that contract rather than applying a separate ulp bound to each component.

`Distribution` validates parameters and holds a `Law`. `Pdf`, `Cdf` and `Quantile` select the law outside their item loops. Formulas use f64 working arithmetic and narrow their outputs. Sampling, rolls and deals take a caller-owned RNG. Keep RNG ownership separate from the numerical algorithms.

### Integers, exact numbers and polynomials

`Overflowing` gives wrapped integer operations and fit flags without branching in the hot loops. `convert::int`/`float` use bit-based conversions where target instructions do not vectorize. Check `NATIVE_64` and `SATURATING` before replacing them with casts. `Divisor` turns repeated division by one integer into multiplication and shifts.

`integer` provides checked gcd, lcm, binomial and power functions. `residue` returns an integer directly, including the dividend for a zero divisor. Objects also have narrow and unsigned forms. `encode`/`decode` handle mixed radix; polynomial decoding handles fields.

`number_theory::Natural` narrows BigUint through u128 to u64 before computing. The fixed-width tiers use Montgomery arithmetic; BigUint uses ordinary residues. Pass cancellation checks by value through narrowing calls. Passing `&mut check` recursively creates new closure types and unbounded monomorphization. `Primes` is a segmented wheel sieve.

`exact` operates on rational numerators/denominators. `linalg::exact` supplies row reduction, solve, pseudoinverse and determinant for fields; least-norm solves use a rank factorization. Permanent uses Ryser's Gray-code formula. Basedpl uses rational matrix solves/determinants and native floating decompositions for approximate inputs. Generic permanent and polynomial helpers also run on f64 and complex floats. Long-running functions expose `_with` cancellation variants.

`polynomial` provides generic `value`, `factored`, `from_roots` and `derivative`, vectorized Horner evaluation, companion-matrix roots and least-squares fitting. Keep Horner's blocks fixed-size: variable slices can introduce copies and iterator setup in downstream codegen. Measure changes in basedpl too.

### Search and gathering

`search` owns exact search, classification, counts, interval index, stable counting/radix sorting and grouped positions. Positions are u32; slices must fit that range. Whole-slice algorithms stay out of line to avoid duplicating them in callers. Each buffer of a `Grouped` holds whole classes. Two classes take a buffer each, which `into_classes` hands over without a copy. More classes share one buffer, because a buffer for each class costs an allocation that callers reading only the positions don't need.

`tolerant` owns relative comparisons, rounding and tolerant searches. Search can index whole-number keys, but validates returned matches unless the haystack is exactly whole and falls back to buckets when needed. Classification intentionally permits tolerant chaining and whole-key groups; it is not the same relation as matching every member against a representative. `Near` and `runs` implement the non-whole paths. Tolerance-sensitive tests need independent inside/outside cases and exact controls.

`gather` copies through a caller's conversion for cycle, repeat, compress, indexed take and scatter. Mask compression skips empty blocks and copies full blocks. In a mixed block, `compress` uses a branch-free loop that reads every item, which suits items loaded from a slice. `positions`, and the split into two classes in `search`, write only the selected items through the 8-bit `SET` table, which suits computed positions. The table loop measured 13% slower for items loaded from a slice. Basedpl reuses these paths for structural operations. Measure search/gather changes there as well.

## Linear algebra

`Inner<T>` objects compile packed matrix products in rustymath. `product` blocks for caches and calls register micro-kernels, with explicit vectors on supported AArch64, AVX2/FMA and wasm SIMD targets. `tiles` owns target tile sizes. Integer products select a narrower exact kernel when their bounds permit it; `Inner::checked` rejects possible overflow. Boolean products use packed bits, and equality-count products use kernels appropriate to their item width.

`Scalar` and `Factor` cover f32/f64 and complex numbers with either parts' width. `Scalar::Real` is the type of singular values and Hermitian eigenvalues. Scalar method names avoid clashes with float traits. Complex division uses scaled Smith arithmetic; public `divide` keeps its faster, narrower-range formula. Complex tiled products use three real products through 3M. Jeremy chose 3M for every complex product. `PlusTimes`' doc gives its accuracy and the fallback for results whose 3M imaginary part isn't finite. Singular/eigenvector signs and phases are not canonicalized.

`linalg/factor.rs` owns LU, Cholesky, QR and their solves. LU factors a transposed copy of the whole matrix, whose columns are contiguous, and `transpose_square` transposes the factors back in place. `lu_panel` halves the columns until `LEAF` or fewer remain. It factors the left half, applies its row swaps to the right half across threads through `swap_rows`, updates the right half through substitution and a packed product, then factors the right half. Cholesky works in blocks of `cholesky_block` columns. Trailing updates and blocked triangular solves use packed products. QR stores Householder blocks in WY form; one `Block::apply` serves QR updates, Q construction and applying Q or its adjoint. QR factors each block of `qr_width` columns through `panel_qr`, which halves the block and updates the right half and T through products. `Block::new` builds T for the eigen and SVD blocks through the same halving. QR pads a working copy whose row length is a multiple of 128 items, as `stride` describes. Cholesky reads only the lower triangle. Thin `qr` and `lstsq` of a tall matrix first try `tall_qr`, CholeskyQR2 with a partial Gram matrix for each thread, above the cutoffs in `tall_for_qr` and `tall_for_lstsq`. The cutoffs admit the sizes where it measured faster on one thread, and larger sizes where its threaded passes are faster on the default thread count. Narrow fits such as 1000000×10 are among the larger sizes, and take up to 1.7 times as long as Householder on one thread. It falls back to Householder where a Cholesky factor fails or `Q₁ᴴQ₁` is far from I. For tall or square matrices, `lstsq` reduces `b` to `Qᴴ b` without storing Q, then solves through R's SVD. Wide matrices solve through their SVD directly.

`solve` and `pinv` use LU when the square matrix's pivots pass the scale-dependent eligibility check. Otherwise they use the least-norm SVD path. The SVD cutoff is shared with `rank`; LU applies no singular-value cutoff. Consequently `rank` can be smaller than the dimensions used by a successful LU solve. Keep this distinction rather than forcing both routes to agree at a rank boundary. `Lu` supplies solves, inverse, factors, determinant and log-determinant. `Lu` and `Svd` are generic over their storage: `LuRef` and `SvdRef` are the borrowed forms, which `new` builds from slices and `as_ref` gives from an owned factorization, with the same methods. `qr_complete` forms full Q.

`factor/eigen.rs` reduces Hermitian matrices to real tridiagonal form. It reads only the lower triangle. Small vector problems use implicit QL; larger ones use divide and conquer with deflation and secular roots. Reconstructing the secular update from the roots maintains orthogonality. Values-only calls omit rotations of eigenvectors. `factor/general.rs` reduces to Hessenberg/Schur form and back-substitutes eigenvectors: real matrices use Francis double shifts, complex matrices use single shifts. Padding avoids cache-set conflicts in column reads. Eigen routines return `None` on failure.

`factor/svd.rs` reduces to a real bidiagonal problem. Small vector problems use Golub-Kahan QR; larger ones use divide and conquer. Values-only calls use QR at every size without constructing or rotating singular vectors. Tall matrices first reduce to R, through `tall_qr` where `qr` or `lstsq` would use it. `truncated_svd` approximates the leading singular values and vectors with a randomized range finder from a fixed seed: products, thin QR factorizations and one small SVD. Power-of-two normalization protects the shift/rotation arithmetic. Nonfinite input or nonconvergence is currently encoded by NaN outputs. Pseudoinverse forms the scaled vector product directly. Source constants own block widths and algorithm cutoffs; test their boundaries through the hidden exports rather than duplicating values in tests.

## Python boundary

A change that adds a user-facing public Rust function also adds its Python binding, following the existing API's names, argument order, dtypes and ufunc or gufunc form.

`py/src/lib.rs` adapts function objects to NumPy ufuncs. `ufuncs!` owns registrations and dtype ordering. Boolean and narrow-integer loops precede wider loops because NumPy chooses by safe casting. Independent callback ranges use `parallel::ranges`, with worker-local `CHUNK` buffers. Stride-zero inputs become `Arg::One`. Checked byte bounds let disjoint arrays and exact in-place mappings split. Other overlaps stay serial. NumPy passes an input that strides ahead of its output without copying it, because one forward pass reads each item before overwriting it. Threads would race on such an input. `Flipped` puts remainder arguments in NumPy order.

Reduction/accumulate calls alias the output with the first input. Supported reductions pass the whole callback lane to `fold_pieces`; accumulations remain serial and carry their working total across chunks. Other operations step scalar forms. Wrapped integer arithmetic/reductions retain their wrapped values. Each callback ends by clearing the calling thread's floating-point flags and raising the flags it started with, plus any explicit failure. The flags it started with keep failures from earlier NumPy callbacks. Nothing reads a worker thread's flags. Tolerant wrappers validate their extra tolerance argument.

`py/src/linalg.rs` registers gufuncs and borrows aligned contiguous core matrices. Strided inputs and f16 inputs use row-major working buffers. Decompositions have float/complex loops; f16 computes in f32. Products also support Boolean and wrapping int64 arithmetic and equality counts. Python decomposition wrappers convert integers/Booleans to float64. Product wrappers preserve Boolean products and promote other integer products to int64. Python `Lu` stores packed factors and row positions; `.l` and `.u` materialize triangular arrays. Its methods and `Svd`'s methods call the borrowed Rust operations without refactoring.

Gufunc signatures cannot express `min(m, n)`, so thin decompositions register separate wide/tall signatures. No input carries the truncated SVD's width, so `svd(a, rank=k)` allocates its outputs and passes them to the `truncated_svd` gufunc as `out`. Singular values and Hermitian eigenvalues have the real dtype; general eigen results are complex; positions/counts are integers. Failures fill NaN and raise NumPy's invalid flag. Preserve the intentional API differences: `svd` returns V, not NumPy's V-adjoint, and complete QR is selected with `complete=True`.

Each gufunc is a unit type whose `Gufunc::each` reads and writes one matrix through a `Cell`. `Cell::read` and `Cell::write` take an argument's position and find its shape and strides in the `MatrixLoop` that registration builds from the signature and dtypes. They also check the argument's dtype and whether it is an input or an output. One `looped` entry point runs every gufunc. It splits a stack across threads through `parallel::stack_with` where byte bounds prove that no two matrices write the same bytes, and otherwise runs the matrices in turn. The check covers only outputs, because NumPy copies any input that an output overlaps before it calls a gufunc loop. NumPy passes an output that overlaps itself or another output unchanged, and such a stack runs in turn.

`py/src/exact.rs` converts flat integer/Fraction inputs to BigRational once and returns Fraction lists. `python/rustymath/exact.py` handles broadcasting and object-array shapes. Exact matrix calls take one 2-D matrix; solve takes a vector or matrix RHS. The ordinary linalg module also exposes permanent and 2-D row reduction. Both row-reduction APIs return `Echelon(rows, pivots, determinant)`.

The remaining binding files match their Python modules. Polynomial wrappers prepare contiguous real or complex arrays; generic f16 operations widen to f32. Integer mixed-radix conversion checks int64 overflow. Number theory takes Python integers through BigUint and polls for Ctrl-C. `search.py` supplies sorting, insertion searches, classification and grouping, with the existing search functions reexported at the package root. Sorts compare exactly, equate signed zeros and put NaNs last. Distribution constructors live in a submodule to avoid collisions with ufunc names.

Complex sums/products and Boolean logical reductions use the shared Rust folds and scans. `minmax` is a gufunc over one axis; `logical_scan` takes a 2×2 truth table. Python exposes `set_threads`, `thread_setting` and `threads`. NumPy controls the one-dimensional runs supplied to legacy callbacks; buffered calls can be too short to split even when the whole array is large.

## Tests and documentation

Rust tests use the public API in `tests/accuracy.rs`, `slices.rs`, `linalg.rs`, `numeric.rs` and `speed.rs`. The library sets `test = false` to avoid a separate `cfg(test)` build. Shared `support::random` and `seeded` use SplitMix64 without additional rand features. Other helpers stay local. Combine related checks into readable tests rather than splitting every assertion into a test.

Python tests are grouped by topic in `tests/test_ufuncs.py`, `test_linalg.py`, `test_search.py` and `test_exact.py`. A test that checks threaded paths takes the `threads` fixture from `tests/conftest.py`, which runs it on one thread and on four.

Elementary real accuracy checks use a four-ulp bound with an additional allowance for f64 libm references. Gamma and incomplete functions use scaled errors and function-specific bounds; f64 iterative checks have a wider allowance. `tests/accuracy.rs` defines those bounds. Real checks cover fused/unfused arithmetic, f16 inputs, sampled wider inputs and reference files. Complex checks exercise public scalar/slice forms at the build's FMA mode, including branch cuts. `testdata/references.py` generates the mpmath references; preserve their notices and `THIRD-PARTY.md`.

`nbs/index.ipynb` generates the README and home page. `src/lib.md` supplies the Rust documentation and runnable doctests. The index includes it through Quarto. Edit those sources, not the generated README. `nbdev-test` checks notebook code; `nbdev-preview` serves saved outputs because `_quarto.yml` disables execution. `.github/workflows/docs.yml` deploys the site.

## CI and release

`Test` runs on pushes to main and pull requests. It uses the shared test-profile build before Python/notebook tests. `Speed` is manual. `Release` runs separately on `v*` tags, builds dist wheels/sdist, publishes the crate to crates.io and the Python artifacts to PyPI, and attaches artifacts to the GitHub release. It does not wait for `Test` or run tests itself.

Before release approval, verify successful tests for the release commit and run `cargo develop && pytest -q` for the installed extension. Confirm `[workspace.package].version` in `Cargo.toml`; Python gets it through `dynamic = ["version"]`. After Jeremy approves, run `ship-release`. It pushes the current version's tag to start publication, then bumps Cargo's version. Never commit, push, tag or publish without approval.
