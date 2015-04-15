#![crate_type="lib"]
#![feature(core)]
#![feature(zero_one)]

// public exports
pub use matrix::{Matrix, zeros, ones, identity, dot, sum};
pub mod matrix;
