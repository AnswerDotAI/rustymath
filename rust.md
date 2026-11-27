

# Using rustymath from Rust

Add rustymath to your `Cargo.toml`:

``` toml
[dependencies]
rustymath = "0.1"
```

On x86-64, build for a recent CPU, with `-C target-cpu=x86-64-v3` or
`-C target-cpu=native`. You can set it once in `.cargo/config.toml`:

``` toml
[target.'cfg(target_arch = "x86_64")']
rustflags = ["-C", "target-cpu=x86-64-v3"]
```

Rust’s default x86-64 target has no fused multiply-add and no vector
instruction for `floor` or `trunc`. Without the setting, rustymath
multiplies and adds separately. Some of its loops also run one item at a
time.

Free functions such as `exp` are generic over `f32`, `f64` and
`half::f16`. The compiler inlines such a function into the loop that
calls it, and vectorizes that loop:

``` rust
use half::f16;
use rustymath::exp;

let e64 = exp(1.0f64); // 2.718281828459045
let e32 = exp(1.0f32); // 2.718282
let e16 = exp(f16::from_f32(1.0)); // 2.719
```

For whole arrays, resolve a formula object such as `Pow` to an
`operation::Kernel`. `Kernel::map` reads each argument through a
`ReadPlan` and replaces the contents of an output `Buffer`. Here `map`
reads `x` in order through `ReadPlan::input`, and repeats the exponent’s
one item through `ReadPlan::copy`:

``` rust
use rustymath::{
    Pow, agreement::{Binding, Mapping, Missing, Read, ReadPlan}, buffer::{Buffer, Element},
    domain::Variant, operation::{Kernel, Settings}, tolerant::Tolerance,
};

let x = [1.5, 2.0, 3.0];
let exponent = [2.0];
let sources = [Binding::full(f64::view(&x)), Binding::full(f64::view(&exponent))];
let plans = [ReadPlan::input(0, 3), ReadPlan::copy(1, 1, Mapping::Single, 3, Missing::Default)];
let reads = plans.each_ref().map(|plan| Read { sources: &sources, plan });
let pow = Kernel::resolve(Pow, Variant::checked(f64::DTYPE), [f64::DTYPE; 2]).unwrap();
let settings = Settings { tolerance: Tolerance::EXACT };
let mut squares = Buffer::with_capacity(pow.result_dtype(), 3);
pow.map(&reads, 0..3, settings, &mut squares).unwrap();
assert_eq!(f64::vec(squares).unwrap(), [2.25, 4.0, 9.0]);
```

`facts::Formula` defines each formula object’s numerical rules,
supported kinds, validity, ranges and derivatives. `Kernel::resolve`
selects the kernel’s computing and result dtypes. `Kernel::of` selects a
formula from a runtime `facts::Op`. Kernels use native bulk routines
where a formula supplies them, and otherwise run its item formula. Their
`map`, `fold_read`, `fold_pieces` and `scan` methods take shared
`agreement::Read` plans. The kernel methods `real`, `integers`,
`complex` and `exact` evaluate scalar arguments with the kernel’s
numerical policy and return a `Result`.

``` rust
use rustymath::{Exp, domain::Variant, operation::Kernel, buffer::{DType, FloatWidth, Width}};

let double = DType::Float(FloatWidth::F64);
let exp = Kernel::resolve(Exp, Variant::checked(double), [DType::Integer(Width::I32)]).unwrap();
assert_eq!(exp.variant().dtype, double);
```

`Sin`, `Cos` and `Tan` use a short argument reduction and polynomials
for arguments up to 2^23 at f64, or 2^20 at f32. An object’s repair
predicate marks arguments of larger magnitude. Its native slice driver
applies the object’s scalar repair routine to those items before
returning the results. `Kernel::ranged` omits repair when the same
predicate is proved false over the supplied ranges.

`atan2(y, x)` and a kernel of `Atan2` give the angle of the point (x,
y), from -π to π. `y` comes first, as in `f64::atan2`.

`divide(x, y)` divides complex numbers and is inlined into the caller’s
loop, like the other free functions. When both arguments are real,
`divide` divides them as reals: a real zero divisor gives ±∞ or NaN,
with an imaginary part of 0. The |y|² in its formula overflows when |y|
is beyond about 1e154 at f64 and 1e19 at f32.

## Tolerant comparison

The `tolerant` module has APL’s tolerant comparison: x equals y when
|x - y| is less than the tolerance times the larger of |x| and |y|. Its
formula objects, such as `Equal`, `Less`, `Floor`, `Residue` and `Gcd`,
hold a `Tolerance`. `operation::Settings` supplies the tolerance for a
kernel execution. With `Tolerance::EXACT`, a kernel chooses ordinary
comparison once for the whole call. A comparison kernel produces Boolean
array results; its scalar `real` method represents true and false as 1
and 0:

``` rust
use rustymath::{
    buffer::Element, domain::Variant, operation::{Kernel, Settings}, tolerant::{Equal, Tolerance},
};

let equal = Kernel::resolve(Equal(Tolerance::DEFAULT), Variant::checked(f64::DTYPE), [f64::DTYPE; 2]).unwrap();
let settings = Settings { tolerance: Tolerance::DEFAULT };
let same = [0.3, 0.1 + 0.2, 0.31].map(|x| equal.real(&[x, 0.3], settings).unwrap());
assert_eq!(same, [1.0, 1.0, 0.0]);
```

## Other modules

- `number_theory`: primality, factorisation, next and previous primes,
  Euler’s totient, and the enumeration and counting of primes, on a
  `u64`, a `u128` or a `BigUint`. Each function computes in the
  narrowest of those types that holds the value. A function that can run
  for a long time checks for an interrupt between steps. `factorial`
  gives n! as a `BigUint`.
- `optimize`: `minimize` finds the minimum of a function of an `f32` or
  `f64` slice. It uses Brent’s method for one variable, Nelder–Mead,
  Powell’s conjugate directions, L-BFGS with a gradient, or differential
  evolution within bounds. `solve` finds a zero or a least-squares
  solution of a function that returns a slice. It uses Brent’s method
  for one equation in one variable and Levenberg–Marquardt otherwise.
  `solve_linear` solves a linear system or finds its least-squares
  solution using only products of the matrix with vectors. It uses
  conjugate gradients, MINRES or restarted GMRES, with an optional
  preconditioner. `solve_affine` uses the same methods to solve f(x) = 0
  for an affine function, using only the function’s values.
  `Method::choose` picks a method when the caller does not specify one.
  Functions implement `Objective` or `Mapping`. They can be closures
  that return a `Result`, whose error stops the search, or types that
  also provide a gradient, vector–Jacobian products or batches of
  points.
- `polynomial`: evaluation over a slice by Horner’s rule, roots from the
  eigenvalues of the companion matrix, building from roots, derivatives
  and least-squares fitting. Roots are sorted by real part, then
  imaginary part. `value` and `factored` evaluate at one point. `value`,
  `factored`, `from_roots` and `derivative` take any type with the field
  operations, such as exact rationals. So do `decode`, which gives the
  values of digits in a mixed radix, and `encode`, which gives the
  digits of values.
- `reduce`: sums, products, maximums and minimums over a slice, and
  their running versions. `Reduction`, implemented by `Sum`, `Product`,
  `Max` and `Min`, defines each reduction’s unit, step for one more
  item, fold of a slice, merge of the folds of two neighbouring runs,
  and scan. Its `checked` method takes one step and reports whether the
  result fits. Float sums and products vectorize by combining the items
  in any order. Integer sums and products, running ones included, wrap
  where they don’t fit and report whether they fit. `extremes` gives the
  least and the greatest item in one pass. `Alternated<Sum>` and
  `Alternated<Product>` are the right folds by `-` and `÷`, written as
  reductions so that every function that folds by a `Reduction` accepts
  them. The right fold by `-` gives the sum of the items at even
  positions less the sum of those at odd positions. The right fold by
  `÷` gives the product of the items at even positions divided by the
  product of those at odd positions. Each total records whether its
  count of items is even or odd. `Totals` drops that record when it
  stores the totals in a buffer. For floats, `Alternated<Product>`
  reports that the result doesn’t fit when a running quotient leaves the
  normal range. Booleans fold by `Max` as `or`, by `Min` as `and`, by
  `Xor` as not-equal, by `Xnor` as equal, and by `SumAs` as a count of
  trues. `Logical` scans Booleans by any function of two Booleans, given
  as its truth table. `SumAs<C>` also sums integers through the traits
  `Term` and `Tally`. The sum wraps in the type `C`, which the caller
  picks to hold every sum: a byte, `i16`, `i32` or `i64`. `along` and
  `scan_along` fold and scan along the middle axis of a row-major array
  of shape `[outer, len, inner]`. For example, `[1, rows, columns]`
  folds or scans down the columns of a matrix. `fold_into` folds the
  same array one block at a time, for a caller that computes the array
  in blocks. Each row length under 16, such as an image’s 3 channels,
  has its own loop. A running maximum or minimum of integers writes its
  current value across each block of 256 items that doesn’t change it.
  `along_mapped` folds an array defined by a mapping over storage,
  without building the array. `sliding` folds each window of
  neighbouring rows. `picks` and `view_picks` give, for each window,
  lane or running fold, the row of the first item that attains its
  maximum or minimum.
- `linalg`: inner products of row-major matrices through the `Inner`
  trait. These include the matrix product `PlusTimes`, which also takes
  complex matrices; the max-plus and min-plus products `MaxPlus` and
  `MinPlus`; the counts of equal and unequal items `PlusEqual` and
  `PlusNotEqual`; and the Boolean products `OrAnd` and `AndOr`. Each
  numeric pair also takes `i64` matrices, whose arithmetic wraps.
  `Inner::checked_into` gives `false` where an integer result doesn’t
  fit. `Product` selects an inner-product pair at run time.
  `Product::run` computes its products through the kernel for the
  arguments’ dtype. The inner-product routines process matrices in
  blocks sized for the caches, as GotoBLAS and BLIS do. Their innermost
  loops use explicit vector instructions on AArch64, in wasm and on
  x86-64 with AVX2. `linalg`’s decompositions of real and complex
  matrices are `lu`, `cholesky`, `qr`, `qr_complete` and `svd`, at
  `f32`, `f64` and `Complex` with parts of either type. It also has
  `Lu::solve`, `Lu::inverse`, `Lu::lower`, `Lu::upper`,
  `Lu::determinant`, `Lu::log_determinant`, `Svd::solve`,
  `Svd::pseudoinverse`, `singular_values` and the least-squares solver
  `lstsq`. `solve` and `pinv` give the least-norm solution and the
  pseudoinverse of any matrix, singular or not. `rank` counts the
  singular values above `lstsq`’s cutoff. `eigh` and
  `hermitian_eigenvalues` take Hermitian matrices, and `is_hermitian`
  tests whether a matrix is Hermitian. `eig` and `eigenvalues` take any
  square matrix. `from_svd`, `from_cholesky`, `from_eigen` and `from_lu`
  multiply a decomposition’s factors back into its matrix.
  `transpose_into` transposes a matrix through 8×8 tiles, applying a
  conversion to each item. `Scalar` is the sealed trait implemented by
  the four matrix item types. `linalg::exact` works in any field, such
  as exact rationals: `echelon`, `solve`, `pinv`, `determinant` and
  Ryser’s `permanent`.
- `exact`: `residue`, `gcd`, `lcm` and `binom` of exact rationals over
  any integer type, such as `BigRational`.
- `distribution`: sixteen distributions. The objects `Pdf`, `Cdf` and
  `Quantile` hold a distribution and evaluate it over a slice. Each
  distribution also samples from a seeded generator.
- `array`: `Array`, a `Layout` paired with the `Buffer` that holds one
  item for each of its positions, in row-major order.
- `create`: filled arrays and integer or floating ranges, evenly spaced
  values and logarithmically spaced values.
- `stats`: means, two-pass variance and standard deviation, weighted
  averages, elementwise p-norms, covariance, correlation and histograms.
  `Axes` prepares reductions over selected axes. Complex variance and
  norms return real values.
- `search`: exact search and classification, stable counting/radix
  sorting, partitioning and quantiles. `select` moves the items of the
  requested ranks to their sorted positions without sorting the items
  between them. Its functions that take a `View` support the same dtypes
  as BasedPL and the Python package. Ordering compares floats exactly,
  treats signed zeros as equal and puts NaNs last in ascending order.
- `buffer`: compact buffers. `Buffer` holds items of one element type,
  and `View` borrows them. `DType` names the type: Booleans, integers of
  four widths (`Width`), floats of three (`FloatWidth`), complex numbers
  of two (`ComplexWidth`), or characters. `DType::common` gives the
  dtype that NumPy promotes two dtypes to, and `DType::inexact` the
  dtype that a function of floats computes in. `IntBuf`, `FloatBuf` and
  `ComplexBuf` each hold items of one family, at any of its widths.
  Their views are `Ints`, `Floats` and `Complexes`. The macros
  `with_ints!`, `with_floats!` and `with_complexes!` run code on a
  view’s slice. `with_width!`, `with_float_width!` and
  `with_complex_width!` run code on a width’s element type.
  `with_dtype!` runs code on a `DType`’s element type. `Element` defines
  each element type’s fill. Its `of`, `view` and `buffer` methods
  convert between slices of that type and a `View` or a `Buffer`.
  `Source` reads an item of one type as another.
- `error`: array functions return `Error`, which contains a message and
  a `Kind`: Length, Rank, Index, Limit, Domain, Overflow, Interrupt or
  Timeout.
- `interrupt`: the process’s interrupt check. A frontend installs one
  with `install`. Long computations call `check` between steps, about
  every `BETWEEN_CHECKS` items, and stop if it returns an Interrupt or
  Timeout error. Until a frontend installs a check, `check` always
  succeeds.
- `memory`: the process’s memory guard. When a binary registers `Guard`
  as its global allocator, Rust aborts the process if an allocation
  would take the total size of its large allocations past the process’s
  limit. The limit defaults to half of physical memory and can be
  changed with `set_limit`. Before an allocation,
  `layout::generated_len` calls `admit`, which returns a Limit error if
  the array would pass the limit.
- `layout`: array shapes, axis keys and names, and their limits. `Shape`
  holds an array’s axis lengths. `Keys` names the positions on one axis.
  `Layout` pairs a shape with keys and a name for each axis.
  `Layout::axis_named` finds an axis by its name. `position` converts an
  index to a position within an axis of a given length, counting from
  the end when the index is negative. `Lanes` describes an array’s items
  as lanes along one axis. It provides each item’s `offset` and the
  `dims` that rustymath’s axis functions take. `element_count` and
  `generated_len` count an array’s items and enforce the limits of
  `MAX_RANK` axes and `MAX_GENERATED_ELEMENTS` items. `generated_len`
  also enforces the memory guard’s limit. `strides` gives the distance
  between neighbouring items along each axis.
- `agreement`: broadcasting by BasedPL’s rule. `Agreement::new` pairs
  the axes of two layouts: axes with the same name first, then the rest
  from the front. If every axis of the left layout pairs and the right
  layout has more axes, the result uses the right layout’s axis order.
  Otherwise, the left layout’s axes come first. An axis of length 1
  repeats along its partner. If both layouts have keys for an axis, the
  result uses the union of those keys. The result is the paired layout
  and a `Mapping` for each argument, which gives the argument’s offset
  for each result position. `blocks` runs a kernel over two slices using
  an agreement, passing contiguous runs whole. `expanded` reads one
  argument’s items for a range of result positions. `Agreement` is
  generic over its layout type, so callers can use `map_layout` to carry
  their own data on layouts. `Frames` pairs the frames of any number of
  arguments to a function that operates on their trailing axes, such as
  matrices in a stack. For each result cell, it gives the corresponding
  cell of each argument.
- `random`: uniform floats, bounded integers and sampling without
  replacement. `roll`, `units`, `below` and `deal` draw through a
  caller’s mutable `rand::Rng`.
- `gather`: copying a slice’s items into a buffer by position. It
  provides `cycle`, `repeat`, `positions` set in a mask, `take` at
  indices, `windows`, `walk` along a pattern of `Steps` with fills, and
  `scatter` at offsets. Each applies the caller’s conversion to each
  item. `walk` also takes `Same`, which copies whole slices of one type.
  `Steps::reach` gives the source offset of one item of a walk.
  `Steps::run` gives the source offset of a run of items that lie in
  order in the source. `Steps::through` composes two walks into one.
  `Steps::per_axis` gives a walk one entry for each axis of its result.
  `Steps::spreads` tells whether an entry repeats an offset or gives a
  fill, and `Steps::rereads` whether two entries walk one source axis,
  as overlapping windows do.
- `structure`: plans for structural operations. A `Plan` holds the
  result’s `Layout` and the `Steps` that `walk` follows to copy the
  result’s items from the source. `select` applies independent axis
  selectors for positions, ranges, index arrays and inserted axes. Index
  arrays are paired in one of two ways: all of them by broadcasting, as
  BasedPL’s `.[]` reads them, or only those that share an axis name, as
  BasedPL’s `⌷` reads them. `addressed` gives the axis that each
  selector selects along. `take` takes or drops items. `rotate` rotates
  or reverses an axis, using one count or one count for each lane.
  `transpose` sends axes to new positions. `catenate` gives the layout
  of two arrays joined along an axis. `Windows` plans windows over
  leading axes, centred on each position of a padded axis. An `Edge`
  mode sets how the plan reads past the axis’s ends. `Walk` describes
  how a copy walks its source, giving the source axis that each entry
  walks. For the entries along one source axis, `read` and `range` give
  what they read as windows: where the windows start, and each entry’s
  length and move. `Walk::windows` recognises a copy of windows laid out
  by `Windows::plan` without padding. It gives each axis’s frames,
  window size and move. `Transpose` gives the kernel that adds a copy’s
  cotangent back to its source: a sum along its spread axes, then reads
  that send each item back, overlap-add for windows that move by more
  than 1, or a scatter-add. Each plan checks its arguments. In each
  plan’s result, axis keys and names stay with their positions.
- `signal`: correlations and convolutions, which slide a kernel along an
  array’s leading axes, and overlap-add. `correlation` gives the plans
  used to read a correlation: the array with the fills added by its
  `Mode`, the array’s windows, and the kernel. For a convolution, the
  kernel is reversed. `correlate` runs these plans through an `Inner`
  pair’s `inner_mapped`. Float and complex `PlusTimes` read the windows
  a block at a time. `overlap_add` adds each frame back where its window
  was read. This is the transpose of taking windows.
- `convert`: `int` converts a whole float to an integer type. It gives
  `None` outside the type’s range, and for `i64` beyond a magnitude of
  2^51 - 1. `float` converts an integer to the nearest float. Loops that
  call either vectorize on every target. AVX2 and wasm lack a vector
  instruction for conversion between `f64` and 64-bit integers. On those
  targets a loop that converts with `as` runs one item at a time.

Like elementary functions, the special functions have free functions and
objects at the crate root. These include `gamma`, `lgamma`, `digamma`,
`erf`, `erfc`, their inverses, `binom` with its slopes `binom_dn` and
`binom_dk`, the incomplete gamma and beta functions and Lambert W.

`Add`, `Subtract`, `Multiply` and the other arithmetic objects at the
crate root implement `Formula`. `Kernel` runs these arithmetic objects
at every float, complex and integer type, using the checking and
wrapping policy specified by its `Variant`. For unbounded rationals,
call `Kernel::exact`, which delegates to `Formula::exact`.

## Accuracy and speed

The table gives nanoseconds per item for each function and for the
standard library’s version of it, measured with a loop over 65,536
inputs on an Apple M5 Max in October 2026. A row named after an object
gives the time of its bulk kernel. The rows `Pow.one` and `Root.one`
give the time of the scalar formulas, called on one pair of arguments at
a time. To print the table on your own machine, run
`RUSTYMATH_COSTS=1 cargo test --test speed`.

<table>
<thead>
<tr>
<th>Function</th>
<th>f64</th>
<th>std f64</th>
<th>f32</th>
<th>std f32</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>Exp</code></td>
<td>0.68</td>
<td>3.92</td>
<td>0.26</td>
<td>0.91</td>
</tr>
<tr>
<td><code>Expm1</code></td>
<td>0.75</td>
<td>2.70</td>
<td>0.30</td>
<td>3.10</td>
</tr>
<tr>
<td><code>Ln</code></td>
<td>0.84</td>
<td>1.62</td>
<td>0.40</td>
<td>1.39</td>
</tr>
<tr>
<td><code>Log1p</code></td>
<td>1.37</td>
<td>2.31</td>
<td>0.67</td>
<td>1.39</td>
</tr>
<tr>
<td><code>Log2</code></td>
<td>0.80</td>
<td>1.62</td>
<td>0.41</td>
<td>1.27</td>
</tr>
<tr>
<td><code>Log10</code></td>
<td>0.85</td>
<td>1.74</td>
<td>0.40</td>
<td>1.27</td>
</tr>
<tr>
<td><code>Sin</code></td>
<td>0.87</td>
<td>4.95</td>
<td>0.36</td>
<td>4.68</td>
</tr>
<tr>
<td><code>Cos</code></td>
<td>0.91</td>
<td>5.12</td>
<td>0.40</td>
<td>5.01</td>
</tr>
<tr>
<td><code>Tan</code></td>
<td>1.14</td>
<td>2.58</td>
<td>0.45</td>
<td>4.68</td>
</tr>
<tr>
<td><code>Asin</code></td>
<td>0.94</td>
<td>4.55</td>
<td>0.34</td>
<td>3.83</td>
</tr>
<tr>
<td><code>Acos</code></td>
<td>1.03</td>
<td>6.03</td>
<td>0.37</td>
<td>5.15</td>
</tr>
<tr>
<td><code>Atan</code></td>
<td>0.89</td>
<td>6.04</td>
<td>0.34</td>
<td>2.16</td>
</tr>
<tr>
<td><code>Sinh</code></td>
<td>1.26</td>
<td>2.47</td>
<td>0.56</td>
<td>2.37</td>
</tr>
<tr>
<td><code>Cosh</code></td>
<td>0.95</td>
<td>2.27</td>
<td>0.38</td>
<td>2.30</td>
</tr>
<tr>
<td><code>Tanh</code></td>
<td>0.94</td>
<td>3.44</td>
<td>0.37</td>
<td>2.28</td>
</tr>
<tr>
<td><code>Asinh</code></td>
<td>1.50</td>
<td>2.56</td>
<td>0.69</td>
<td>1.87</td>
</tr>
<tr>
<td><code>Acosh</code></td>
<td>1.59</td>
<td>2.17</td>
<td>0.68</td>
<td>1.56</td>
</tr>
<tr>
<td><code>Atanh</code></td>
<td>1.40</td>
<td>3.78</td>
<td>0.62</td>
<td>1.62</td>
</tr>
<tr>
<td><code>Cbrt</code></td>
<td>1.11</td>
<td>1.39</td>
<td>0.89</td>
<td>1.85</td>
</tr>
<tr>
<td><code>Pow</code></td>
<td>2.92</td>
<td>7.28</td>
<td>1.61</td>
<td>2.06</td>
</tr>
<tr>
<td><code>Pow.one</code></td>
<td>4.11</td>
<td>7.18</td>
<td>2.27</td>
<td>1.97</td>
</tr>
<tr>
<td><code>Hypot</code></td>
<td>0.55</td>
<td>1.21</td>
<td>0.27</td>
<td>0.87</td>
</tr>
<tr>
<td><code>Root</code></td>
<td>3.38</td>
<td>4.24</td>
<td>1.76</td>
<td>1.96</td>
</tr>
<tr>
<td><code>Root.one</code></td>
<td>4.23</td>
<td>4.25</td>
<td>2.36</td>
<td>2.09</td>
</tr>
<tr>
<td><code>Maximum</code></td>
<td>0.19</td>
<td>0.19</td>
<td>0.10</td>
<td>0.09</td>
</tr>
<tr>
<td><code>reduce::sum</code></td>
<td>0.06</td>
<td>0.49</td>
<td>0.03</td>
<td>0.46</td>
</tr>
<tr>
<td><code>reduce::max</code></td>
<td>0.06</td>
<td>0.11</td>
<td>0.03</td>
<td>0.05</td>
</tr>
<tr>
<td><code>reduce::Sum::scan</code></td>
<td>0.15</td>
<td>1.94</td>
<td>0.15</td>
<td>2.61</td>
</tr>
<tr>
<td><code>reduce::Max::scan</code></td>
<td>0.15</td>
<td>2.53</td>
<td>0.07</td>
<td>2.54</td>
</tr>
<tr>
<td><code>polynomial::evaluate</code></td>
<td>0.51</td>
<td>2.60</td>
<td>0.22</td>
<td>2.60</td>
</tr>
</tbody>
</table>

Functions on `f16` compute in `f32` and round once. IEEE 754 conformance
for infinities, NaNs and signed zeros is not promised. The tests compare
finite results with references at every `f16` input, at samples of `f32`
and `f64` inputs, and at published hard cases.

Where the target has a fused multiply-add instruction, as on AArch64 and
on x86-64 with FMA, the functions use it. Elsewhere, as in wasm, they
multiply and add separately, because calling a software `fma` would
prevent the compiler from vectorizing the loop. The constant `FUSED` is
true where the target uses fused multiply-add. The modules `fused` and
`unfused` contain every function again, with fused multiply-add always
on or always off. In a loop compiled with FMA for itself, such as an
AVX2 version that your program chooses at run time, call `fused`. Either
way, results stay within 4 ulps of the correctly rounded result, apart
from the [documented
exceptions](https://answerdotai.github.io/rustymath/accuracy.html).

General powers and roots use compensated native block algorithms. At
`f64`, logarithms and exponentials use tables. Exceptional inputs take
the scalar repair path. Cheap constant powers, square roots and cube
roots use specialized formulas.
