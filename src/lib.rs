//! Numerical kernels and value-semantic arrays.
#![doc = include_str!("lib.md")]
#[macro_use]
extern crate rustymath_core;
pub use rustymath_core::*;
pub use rustymath_kernels::operation;
pub mod array;
pub mod create;
pub mod linalg;

