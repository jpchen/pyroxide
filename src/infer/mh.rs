//! Random-walk Metropolis–Hastings with adaptive proposal scale and covariance.
//!
//! The proposal is `z' = z + s * L eps`, `eps ~ N(0, I)`, where `L L^T` is the
//! estimated posterior covariance (diagonal by default; dense with
//! [`dense_mass`](MetropolisHastings::dense_mass)) and `s` is a global scale.
//! During warmup the covariance is estimated with Welford's algorithm in the
//! same windowed schedule NUTS uses, and `s` is tuned by dual averaging toward
//! a target acceptance rate (default 0.234, the Roberts–Gelman–Gilks optimum
//! for high-dimensional Gaussian targets).
//!
//! Metropolis–Hastings only needs the potential's *value*, never its gradient,
//! so models are evaluated with plain `f64` here.

use rand::Rng;
use rand_distr::StandardNormal;

use super::adapt::{AdaptConfig, MassMatrix, WarmupAdapter};
use super::mcmc::{ChainRng, Kernel};
use super::potential::Potential;

/// Random-walk Metropolis–Hastings kernel.
pub struct MetropolisHastings<P: Potential> {
    potential: P,
    /// Initial proposal scale multiplier. Defaults to `2.38 / sqrt(d)`.
    pub step_size: Option<f64>,
    pub adapt: AdaptConfig,
}

impl<P: Potential> MetropolisHastings<P> {
    pub fn new(potential: P) -> Self {
        MetropolisHastings {
            potential,
            step_size: None,
            adapt: AdaptConfig {
                target_accept_prob: 0.234,
                prox_center_multiplier: 1.0,
                restart_step_size_per_window: false,
                ..AdaptConfig::default()
            },
        }
    }
    pub fn step_size(mut self, s: f64) -> Self {
        self.step_size = Some(s);
        self
    }
    pub fn adapt_step_size(mut self, b: bool) -> Self {
        self.adapt.adapt_step_size = b;
        self
    }
    /// Adapt the proposal covariance during warmup (default true).
    pub fn adapt_covariance(mut self, b: bool) -> Self {
        self.adapt.adapt_mass_matrix = b;
        self
    }
    /// Use a dense (full) proposal covariance instead of a diagonal one.
    pub fn dense_mass(mut self, b: bool) -> Self {
        self.adapt.dense_mass = b;
        self
    }
    pub fn target_accept_prob(mut self, p: f64) -> Self {
        self.adapt.target_accept_prob = p;
        self
    }
    pub fn potential_ref(&self) -> &P {
        &self.potential
    }
}

/// Per-chain state of [`MetropolisHastings`].
#[derive(Clone, Debug)]
pub struct MhState {
    pub i: usize,
    pub num_warmup: usize,
    pub z: Vec<f64>,
    pub potential_energy: f64,
    pub accept_prob: f64,
    pub mean_accept_prob: f64,
    pub adapt: WarmupAdapter,
    proposal: Vec<f64>,
    eps: Vec<f64>,
}

impl<P: Potential> Kernel for MetropolisHastings<P> {
    type State = MhState;

    fn potential(&self) -> &dyn Potential {
        &self.potential
    }

    fn init(&self, z: Vec<f64>, num_warmup: usize, _rng: &mut ChainRng) -> MhState {
        let d = z.len();
        let pe = self.potential.value(&z);
        let step_size = self.step_size.unwrap_or(2.38 / (d.max(1) as f64).sqrt());
        let adapt = WarmupAdapter::new(num_warmup, d, step_size, None, self.adapt.clone(), None);
        MhState {
            i: 0,
            num_warmup,
            z,
            potential_energy: pe,
            accept_prob: 0.0,
            mean_accept_prob: 0.0,
            adapt,
            proposal: vec![0.0; d],
            eps: vec![0.0; d],
        }
    }

    fn step(&self, st: &mut MhState, rng: &mut ChainRng) {
        let warmup = st.i < st.num_warmup;
        let s = st.adapt.step_size;
        // proposal = z + s * L eps, where L L^T = estimated covariance
        for e in st.eps.iter_mut() {
            *e = rng.sample(StandardNormal);
        }
        match &st.adapt.mass {
            MassMatrix::Diag { inv, .. } => {
                for i in 0..st.z.len() {
                    st.proposal[i] = st.z[i] + s * inv[i].sqrt() * st.eps[i];
                }
            }
            MassMatrix::Dense { dim, chol_inv, .. } => {
                crate::linalg::tril_matvec(chol_inv, *dim, &st.eps, &mut st.proposal);
                for i in 0..*dim {
                    st.proposal[i] = st.z[i] + s * st.proposal[i];
                }
            }
        }
        let pe_new = self.potential.value(&st.proposal);
        let mut delta = pe_new - st.potential_energy;
        if delta.is_nan() {
            delta = f64::INFINITY;
        }
        let accept_prob = (-delta).exp().min(1.0);
        let u: f64 = rng.random();
        if u < accept_prob {
            std::mem::swap(&mut st.z, &mut st.proposal);
            st.potential_energy = pe_new;
        }
        if warmup {
            let (t, z) = (st.i, &st.z);
            st.adapt.update(t, accept_prob, z, None);
        }
        st.i += 1;
        let n = if warmup { st.i } else { st.i - st.num_warmup };
        st.mean_accept_prob += (accept_prob - st.mean_accept_prob) / n as f64;
        st.accept_prob = accept_prob;
    }

    fn position(st: &MhState) -> &[f64] {
        &st.z
    }

    fn stat_names(&self) -> &'static [&'static str] {
        &["accept_prob", "step_size", "potential_energy", "mean_accept_prob"]
    }

    fn stats(&self, st: &MhState, out: &mut [f64]) {
        out[0] = st.accept_prob;
        out[1] = st.adapt.step_size;
        out[2] = st.potential_energy;
        out[3] = st.mean_accept_prob;
    }
}
