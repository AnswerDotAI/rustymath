//! Matrix algorithms and their array interface.
pub use rustymath_core::linalg::*;
mod stacked;
pub use stacked::{Stacked, cells, paired, stored};

