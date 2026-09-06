//! Inference algorithms and the MCMC driver.
//!
//! * [`Potential`] – what every algorithm targets: `U(z) = -log p(z)` with gradient.
//! * [`ModelPotential`] – a [`Model`](crate::model::Model) as a potential.
//! * [`HmcKernel`] – NUTS ([`HmcKernel::nuts`]) and HMC ([`HmcKernel::hmc`]).
//! * [`MetropolisHastings`] – adaptive random-walk Metropolis–Hastings.
//! * [`MCMC`] – warmup + sampling over parallel chains, returning [`Samples`].

pub mod adapt;
pub mod hmc;
pub mod mcmc;
pub mod mh;
pub mod potential;

pub use adapt::{AdaptConfig, MassMatrix, WarmupAdapter};
pub use hmc::{Algo, HmcConfig, HmcKernel, HmcState};
pub use mcmc::{Array, ChainRng, InitStrategy, Kernel, Samples, MCMC};
pub use mh::{MetropolisHastings, MhState};
pub use potential::{FnPotential, ModelPotential, Potential};
