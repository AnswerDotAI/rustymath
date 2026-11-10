//! Numerical kernels and value-semantic arrays.
#![doc = include_str!("lib.md")]
#[macro_use]
extern crate rustymath_core;
// The public surface is math, arrays and their execution facts, not every helper
// made public inside core so the kernels crate can call it.
pub use rustymath_core::{
    agreement, buffer, characters, convert, derivative, distribution, domain, error, events, exact, facts, gather, hashing, integer, interrupt,
    layout, memory, number_theory, optimize, parallel, polynomial, random, recipe, reduce, search, signal, stats, structure, tolerant,
    fused, unfused, half, num_complex,
};
pub use rustymath_core::{
    FUSED, Float, Native, Arg,
    Abs, Acos, Acosh, Add, Angle, Asin, Asinh, Atan, Atan2, Atanh, Betainc, BetaincAt, Betaincinv, Binom, BinomDk, BinomDn,
    BitAnd, BitOr, Bitwise, Cbrt, Choose, Cis, Clip, Conj, Cos, Cosh, Digamma, Divide, Erf, Erfc, Erfcinv, Erfinv, Exp, Expm1,
    Factorial, Gamma, Gammainc, Gammaincc, Gammaincinv, Hypot, Imag, Isfinite, Isinf, Isnan, LambertW, LeftShift, Lgamma, Ln, Log1p, Log2,
    Log10, Logb, MathishDivide, MathishLogb, Multiply, Nand, Negate, Nor, Not, PiTimes, Polygamma, Pow, Real, RealAcos, RealAcosh, RealAsin,
    RealAtanh, RealLn, RealPow, RealSqrt, RealSqrt1mx2, RealSqrtm1mx2, RealSqrtx2m1, Reciprocal, RightShift, Rint, Root, Round, Scale,
    Sign, Sin, Sinh, Sqrt, Sqrt1mx2, Sqrt1px2, Sqrtm1mx2, Sqrtx2m1, Subtract, Tan, Tanh, Trunc, Exponent,
};
pub use rustymath_core::{
    acos, acosh, asin, asinh, atan, atan2, atanh, beta, betainc, betaincinv, binom, binom_dk, binom_dn, bitwise, cbrt, cos, cosh, digamma,
    divide, erf, erfc, erfcinv, erfinv, exp, expm1, factorial, gamma, gammainc, gammaincc, gammaincinv, hypot, lambertw, lgamma, ln, log1p,
    log2, log10, logb, polygamma, pow, root, sin, sinh, sqrt1mx2, sqrt1px2, sqrtx2m1, tan, tanh,
};
pub use rustymath_core::{
    element_types, function_doc, native_reduction, with_complex_width, with_complexes, with_dtype, with_float_width, with_floats, with_inexact,
    with_int_pair, with_ints, with_reduce, with_tolerance, with_width,
};
pub use rustymath_kernels::operation;
pub mod array;
pub mod create;
pub mod linalg;
