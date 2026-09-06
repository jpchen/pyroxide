//! # pyroxide
//!
//! A fast, decoupled probabilistic programming library for Rust.
//!
//! * [`ad`] – reverse-mode automatic differentiation (`f64` / [`ad::Var`]).
//! * [`dist`] – probability distributions with analytic gradients.
//! * [`model`] – the effect-handler layer: write a generative model once, run it
//!   under different handlers (trace, condition, log-density).
//! * [`infer`] – MCMC: NUTS, HMC, random-walk Metropolis–Hastings, warmup
//!   adaptation, and the [`infer::MCMC`] driver.
//! * [`diagnostics`] – effective sample size, R-hat, HPDI, summaries.
//!
//! See `docs/DESIGN.md` in the repository for the design rationale.

pub mod ad;
pub mod dist;
pub mod linalg;
pub mod model;
pub mod special;

pub use ad::{Real, Var};

/// Convenient glob import for writing models and running inference.
pub mod prelude {
    pub use crate::ad::{Real, Var};
    pub use crate::dist::*;
    pub use crate::model::{Handler, Model};
}
