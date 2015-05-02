#![crate_type="lib"]

extern crate num;
// public exports
pub use matrix::{Matrix, zeros, ones, identity, dot, sum};
pub mod matrix;
