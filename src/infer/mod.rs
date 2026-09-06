//! Inference algorithms and the MCMC driver.
//!
//! * [`Potential`] – what every algorithm targets: `U(z) = -log p(z)` with gradient.
//! * [`ModelPotential`] – a [`Model`](crate::model::Model) as a potential.
//! * [`HmcKernel`] – NUTS ([`HmcKernel::nuts`]) and HMC ([`HmcKernel::hmc`]).
//! * [`MetropolisHastings`] – adaptive random-walk Metropolis–Hastings.
//! * [`BarkerMH`] – gradient-based Barker proposal (robust, low/moderate dimension).
//! * [`AIES`], [`ESS`] – gradient-free ensemble samplers (affine-invariant / slice).
//! * [`MAMS`] – Metropolis-adjusted microcanonical (isokinetic Langevin) sampler.
//! * [`MCMC`] – warmup + sampling over parallel chains, returning [`Samples`].

pub mod adapt;
pub mod barker;
pub mod ensemble;
pub mod hmc;
pub mod mams;
pub mod mcmc;
pub mod mh;
pub mod potential;

pub use adapt::{AdaptConfig, MassMatrix, WarmupAdapter};
pub use barker::{BarkerMH, BarkerState};
pub use ensemble::{AiesMove, EnsembleState, AIES, ESS};
pub use hmc::{Algo, HmcConfig, HmcKernel, HmcState};
pub use mams::{MamsState, MAMS};
pub use mcmc::{Array, ChainRng, InitStrategy, Kernel, Samples, MCMC};
pub use mh::{MetropolisHastings, MhState};
pub use potential::{FnPotential, ModelPotential, Potential};
