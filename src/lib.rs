#![doc = include_str!("../README.md")]
mod exp;
mod float;
mod ln;
mod trig;
pub mod tolerant;
#[cfg(test)]
mod accuracy;

pub use exp::exp;
pub use float::{Float, Native, FUSED};
pub use ln::ln;
pub use trig::{checked_cos, checked_sin, cos, sin};
